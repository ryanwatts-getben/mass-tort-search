import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone  # Update import
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

def load_kidney_cancer_criteria():
    """Load kidney cancer criteria from txt file."""
    try:
        with open('src/kidney_cancer.txt', 'r') as file:
            content = file.read()
        
        # Parse the content into categories
        criteria = {}
        current_category = None
        
        for item in content.split(','):
            item = item.strip()
            if item in ['TEST', 'PROBLEM', 'TREATMENT', 'Symptoms', 'LabResultsRedFlags', 'DiagnosticProcedures', 'TreatmentOptions']:
                current_category = item
                criteria[current_category] = []
            elif item and current_category:
                # Skip ICD codes (items that contain numbers)
                if not any(char.isdigit() for char in item):
                    criteria[current_category].append(item.lower())
        
        return criteria
        
    except Exception as e:
        logger.error(f"Error loading kidney cancer criteria: {str(e)}")
        return {}

def search_medical_records():
    """Search medical records using kidney cancer criteria."""
    try:
        # Initialize Pinecone with GRPC
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        
        # Load kidney cancer criteria
        criteria = load_kidney_cancer_criteria()
        
        # Initialize results list
        all_results = []
        
        # Search terms to look for in metadata
        search_fields = {
            'test': criteria.get('DiagnosticProcedures', []) + criteria.get('LabResultsRedFlags', []),
            'problem': criteria.get('Symptoms', []) + criteria.get('PROBLEM', []),
            'treatment': criteria.get('TreatmentOptions', []) + criteria.get('TREATMENT', []),
        }
        
        # Search for each term in each field
        for field, terms in search_fields.items():
            for term in terms:
                try:
                    # Format the term to match the flattened entity structure
                    formatted_term = f"{field}:{term}"
                    
                    # Create metadata filter for entities field
                    filter_dict = {
                        'entities': {"$in": [formatted_term]}
                    }
                    
                    # Query Pinecone with GRPC client
                    query_response = index.query(
                        vector=[0] * 768,  # Dummy vector since we're only using metadata filter
                        filter=filter_dict,
                        top_k=100,
                        include_metadata=True,
                        namespace="default"  # Add namespace parameter
                    )
                    
                    # Process results (handle GRPC response format)
                    for match in query_response.matches:  # Note: no ['matches'] indexing needed
                        metadata = match.metadata
                        result = {
                            'case_id': metadata.get('case_id', 'unknown'),
                            'document_id': metadata.get('document_id', 'unknown'),
                            'page_number': metadata.get('page_number', 'unknown'),
                            'text': metadata.get('text', '')[:200],
                            'search_term': term,
                            'field_matched': field,
                            'score': match.score  # Note: direct attribute access
                        }
                        all_results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error searching for term '{term}' in field '{field}': {str(e)}")
                    continue
        
        # Convert results to DataFrame and remove duplicates
        df_results = pd.DataFrame(all_results)
        df_results = df_results.drop_duplicates(subset=['case_id', 'document_id', 'page_number', 'search_term'])
        
        # Sort by score
        df_results = df_results.sort_values('score', ascending=False)
        
        # Save results to CSV
        output_file = 'kidney_cancer_results.csv'
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Also save detailed results to JSON
        detailed_output = 'src/kidney_cancer_results.json'
        df_results.to_json(detailed_output, orient='records', indent=2)
        logger.info(f"Detailed results saved to {detailed_output}")
        
        return df_results
        
    except Exception as e:
        logger.error(f"Error in search_medical_records: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to execute the search."""
    try:
        logger.info("Starting kidney cancer medical records search")
        results = search_medical_records()
        
        if not results.empty:
            logger.info(f"Found {len(results)} matching records")
            
            # Print summary of results
            print("\nSearch Results Summary:")
            print("-" * 80)
            for _, row in results.iterrows():
                print(f"Case ID: {row['case_id']}")
                print(f"Document ID: {row['document_id']}")
                print(f"Page Number: {row['page_number']}")
                print(f"Matched Term: {row['search_term']} ({row['field_matched']})")
                print(f"Score: {row['score']:.3f}")
                print(f"Text Preview: {row['text'][:100]}...")
                print("-" * 80)
        else:
            logger.warning("No matching records found")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
