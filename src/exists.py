import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging
from utils import load_config, setup_logging
import pandas as pd

# Set up logging
logger = setup_logging()

# Load configuration
config = load_config('config/config.yaml')
index_name = config['pinecone']['index_name']

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
pc = Pinecone(api_key=PINECONE_API_KEY)

def check_metadata_fields():
    """Check which metadata fields exist in the Pinecone index."""
    try:
        # Initialize Pinecone
        index = pc.Index(index_name)

        # Define all possible metadata fields (updated with new keys)
        metadata_fields = [
            'test', 'problem', 'treatment', 'drug', 'anatomy',
            'syntactic_relationships', 'custom_relationships',
            'text', 'case_id', 'document_id', 'page_number'
        ]

        field_existence = {}
        field_counts = {}

        # Check each field
        for field in metadata_fields:
            try:
                # Create filter to check if field exists
                filter_dict = {
                    field: {"$exists": True}
                }

                # Query Pinecone with the filter
                query_response = index.query(
                    vector=[0] * 768,  # Dummy vector for metadata-only search
                    filter=filter_dict,
                    top_k=1,  # We only need to know if any records exist
                    include_metadata=True
                )

                # Field exists if we got any matches
                exists = len(query_response['matches']) > 0
                field_existence[field] = exists

                if exists:
                    # Get count of records with this field
                    count_response = index.query(
                        vector=[0] * 768,
                        filter=filter_dict,
                        top_k=10000,  # Adjust based on your needs
                        include_metadata=False
                    )
                    field_counts[field] = len(count_response['matches'])
                else:
                    field_counts[field] = 0

            except Exception as e:
                logger.error(f"Error checking field {field}: {str(e)}")
                field_existence[field] = False
                field_counts[field] = 0

        # Create results DataFrame
        results = pd.DataFrame({
            'Field': list(field_existence.keys()),
            'Exists': list(field_existence.values()),
            'Count': [field_counts.get(field, 0) for field in field_existence.keys()]
        })

        # Sort by existence and count
        results = results.sort_values(['Exists', 'Count'], ascending=[False, False])

        # Save results
        results.to_csv('metadata_fields_existence.csv', index=False)
        
        return results

    except Exception as e:
        logger.error(f"Error checking metadata fields: {str(e)}")
        return pd.DataFrame()

def print_field_summary(results: pd.DataFrame):
    """Print a summary of metadata field existence and counts."""
    if results.empty:
        print("No results available.")
        return

    print("\nMetadata Fields Summary:")
    print("-" * 80)
    
    # Print existing fields
    print("Existing Fields:")
    existing = results[results['Exists']]
    for _, row in existing.iterrows():
        print(f"  - {row['Field']}: {row['Count']} records")
    
    # Print missing fields
    print("\nMissing Fields:")
    missing = results[~results['Exists']]
    for _, row in missing.iterrows():
        print(f"  - {row['Field']}")
    
    print("-" * 80)
    
    # Print summary statistics
    total_fields = len(results)
    existing_fields = len(existing)
    print(f"Summary:")
    print(f"  Total Fields Checked: {total_fields}")
    print(f"  Fields Present: {existing_fields}")
    print(f"  Fields Missing: {total_fields - existing_fields}")
    print("-" * 80)

def main():
    """Main function to check metadata field existence."""
    try:
        logger.info("Starting metadata field existence check")
        results = check_metadata_fields()
        
        if not results.empty:
            print_field_summary(results)
        else:
            logger.warning("No results obtained from metadata field check")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
