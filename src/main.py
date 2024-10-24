import os
import boto3
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone  # Update import
from concurrent.futures import ThreadPoolExecutor
from data_ingestion import ingest_data_from_s3
from preprocessing import preprocess_data
from batch_processing import process_batch_parallel, process_batches_with_progress
from pinecone_operations import upload_to_pinecone, search_pinecone, index_name
from utils import load_config, load_search_criteria, generate_report, setup_logging
from datasets import Dataset, concatenate_datasets
from vectorization import initialize_vectorization_model
from scoring import score_and_rank_results
import torch

# Set up logging
logger = setup_logging()

# Load configuration
config = load_config('config/config.yaml')

def main():
    try:
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Initialize models once
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        model_name = config['models']['ner']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        vectorization_model = initialize_vectorization_model(device)

        # Initialize S3 client
        s3 = boto3.client(
            's3',
            region_name=config['aws']['region'],
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        # Initialize Pinecone with GRPC client
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Data Ingestion with parallel processing
        bucket = config['aws']['s3']['bucket']
        use_prefixes = config['aws']['s3'].get('use_prefixes', True)
        prefixes = config['aws']['s3'].get('prefixes', [])

        # Process each prefix in parallel if use_prefixes is True
        if use_prefixes:
            with ThreadPoolExecutor() as executor:
                futures = []
                for prefix in prefixes:
                    future = executor.submit(ingest_data_from_s3, s3, bucket, prefix)
                    futures.append(future)

                # Collect results as datasets
                raw_datasets = []
                for future in futures:
                    try:
                        dataset = future.result()
                        if len(dataset) > 0:
                            raw_datasets.append(dataset)
                    except Exception as e:
                        logger.error(f"Error processing prefix: {str(e)}")
                        continue
        else:
            # Process all files in the bucket
            logger.info("Processing all files in the bucket without prefix restriction")
            dataset = ingest_data_from_s3(s3, bucket)
            if len(dataset) > 0:
                raw_datasets = [dataset]
            else:
                logger.error("No data retrieved from S3. Exiting pipeline.")
                return

        # Concatenate datasets
        if raw_datasets:
            raw_dataset = concatenate_datasets(raw_datasets)
            logger.info(f"Combined dataset with {len(raw_dataset)} examples")
        else:
            logger.error("No data retrieved from S3. Exiting pipeline.")
            return

        # Process data using pipeline approach
        processed_dataset = process_batch_parallel(
            raw_dataset,
            device,
            tokenizer,
            ner_model,
            vectorization_model
        )

        if len(processed_dataset) == 0:
            logger.error("No data processed successfully. Exiting pipeline.")
            return

        # Verify that processed_dataset has expected new metadata structure
        # You may include additional checks or logs if necessary

        # Upload to Pinecone
        upload_to_pinecone(processed_dataset, config['pinecone']['index_name'])

        # Search and Score (if needed)
        if config.get('perform_search', False):
            search_criteria = load_search_criteria(config)
            search_results = search_pinecone(search_criteria, config['pinecone']['index_name'])
            final_results = score_and_rank_results(search_results)
            generate_report(final_results)

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
