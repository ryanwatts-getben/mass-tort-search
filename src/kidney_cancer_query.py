import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
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

def load_search_terms():
    """Load ALL comma-separated terms from kidney_cancer.txt file."""
    try:
        with open('src/kidney_cancer.txt', 'r') as file:
            content = file.read()
        
        # Split by comma and clean up terms, keeping everything
        terms = []
        for term in content.split(','):
            term = term.strip()
            # Only skip empty terms, keep everything else including ICD codes
            if term:
                # Convert to lowercase and add to terms
                terms.append(term.lower())
        
        # Remove duplicates while preserving order
        unique_terms = list(dict.fromkeys(terms))
        
        logger.info(f"Loaded {len(unique_terms)} unique search terms:")
        for term in unique_terms:
            logger.info(f"  - {term}")
        
        return unique_terms
        
    except Exception as e:
        logger.error(f"Error loading search terms: {str(e)}")
        return []

def search_medical_records():
    """Search medical records using all terms from kidney_cancer.txt."""
    try:
        # Initialize Pinecone with GRPC
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to index: {index_name}")
        
        # Load search terms
        search_terms = load_search_terms()
        if not search_terms:
            logger.error("No search terms loaded. Exiting search.")
            return pd.DataFrame()
        
        logger.info(f"Searching for {len(search_terms)} terms across all metadata fields")
        
        # Build metadata filter to search across all relevant fields
        metadata_filter = {
            "$or": [
                {"test": {"$in": search_terms}},
                {"problem": {"$in": search_terms}},
                {"treatment": {"$in": search_terms}},
                {"drug": {"$in": search_terms}},
                {"anatomy": {"$in": search_terms}},
                {"text": {"$in": search_terms}}  # Also search in the text field
            ]
        }
        
        logger.info(f"Metadata filter constructed: {metadata_filter}")
        
        # Create a dummy zero vector
        vector_dimension = config['pinecone']['dimension']
        dummy_vector = [0.0] * vector_dimension
        
        # Query Pinecone index with metadata filter
        logger.info("Executing Pinecone query...")
        query_response = index.query(
            vector=dummy_vector,
            filter=metadata_filter,
            top_k=1000,  # Increased to get more matches
            include_metadata=True,
            namespace="default"
        )
        
        logger.info(f"Query completed. Found {len(query_response.matches)} matches.")
        
        # Process results
        all_results = []
        print("\nProcessing matches:")
        print("-" * 80)
        
        for i, match in enumerate(query_response.matches, 1):
            metadata = match.metadata or {}
            print(f"\nMatch {i}:")
            print(f"Metadata: {metadata}")
            
            # Find which terms matched in which fields
            matched_terms = {
                'test': [t for t in search_terms if t in metadata.get('test', [])],
                'problem': [t for t in search_terms if t in metadata.get('problem', [])],
                'treatment': [t for t in search_terms if t in metadata.get('treatment', [])],
                'drug': [t for t in search_terms if t in metadata.get('drug', [])],
                'anatomy': [t for t in search_terms if t in metadata.get('anatomy', [])],
                'text': [t for t in search_terms if t.lower() in metadata.get('text', '').lower()]
            }
            
            # Count total matches
            total_matches = sum(len(terms) for terms in matched_terms.values())
            
            result = {
                'case_id': metadata.get('case_id', 'unknown'),
                'document_id': metadata.get('document_id', 'unknown'),
                'page_number': metadata.get('page_number', 'unknown'),
                'text': metadata.get('text', '')[:200],
                'matched_terms': matched_terms,
                'total_matches': total_matches,
                'score': getattr(match, 'score', None)
            }
            
            # Log match details
            logger.info(f"Processing match {i}:")
            logger.info(f"Case ID: {result['case_id']}")
            logger.info(f"Document ID: {result['document_id']}")
            logger.info(f"Total matched terms: {total_matches}")
            
            all_results.append(result)
        
        if not all_results:
            logger.warning("No matches found.")
            return pd.DataFrame()

        # Convert results to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Sort by total matches and score
        df_results = df_results.sort_values(['total_matches', 'score'], 
                                          ascending=[False, False])
        
        # Remove duplicates
        original_len = len(df_results)
        df_results = df_results.drop_duplicates(subset=['case_id', 'document_id', 'page_number'])
        if len(df_results) < original_len:
            logger.info(f"Removed {original_len - len(df_results)} duplicate records")
        
        # Save results
        output_file = 'kidney_cancer_results.csv'
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        detailed_output = 'src/kidney_cancer_results.json'
        df_results.to_json(detailed_output, orient='records', indent=2)
        logger.info(f"Detailed results saved to {detailed_output}")
        
        # Print summary statistics
        print("\nSearch Results Summary:")
        print("-" * 80)
        print(f"Total matches found: {len(query_response.matches)}")
        print(f"Unique documents after deduplication: {len(df_results)}")
        print(f"Number of cases: {df_results['case_id'].nunique()}")
        print(f"Average matches per document: {df_results['total_matches'].mean():.2f}")
        print("-" * 80)
        
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
            
            # Print detailed results
            print("\nDetailed Results:")
            print("-" * 80)
            for _, row in results.iterrows():
                print(f"Case ID: {row['case_id']}")
                print(f"Document ID: {row['document_id']}")
                print(f"Page Number: {row['page_number']}")
                print(f"Total Matches: {row['total_matches']}")
                print("\nMatched Terms by Category:")
                for field, terms in row['matched_terms'].items():
                    if terms:
                        print(f"  {field}: {terms}")
                print(f"Score: {row.get('score', 'N/A')}")
                print(f"Text Preview: {row['text'][:100]}...")
                print("-" * 80)
        else:
            logger.warning("No matching records found")
            print("\nNo matching records found in the database.")
            print("Please verify:")
            print("1. The index exists and contains documents")
            print("2. The search terms are present in the metadata")
            print("3. The metadata fields contain the expected values")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
