#!/usr/bin/env python3
"""
Test script for abbreviation extraction in TEI processor
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from tei.processor import TEIProcessor
from config import Config

def main():
    # Load configuration
    config = Config('config.toml')
    
    # Initialize the TEI processor
    processor = TEIProcessor(config)
    
    # Parse the test document
    sample_path = Path(__file__).parent / 'samples' / 'abbr_tei_sample.xml'
    logger.info(f"Parsing document: {sample_path}")
    
    abbreviations, tree = processor.parse_document(sample_path)
    
    logger.info(f"Found {len(abbreviations)} abbreviations")
    
    # Display details of each abbreviation
    for i, abbr in enumerate(abbreviations):
        element_type = abbr.abbr_element.tag.split('}')[-1] if '}' in abbr.abbr_element.tag else abbr.abbr_element.tag
        ref = abbr.abbr_element.get('ref', '') if element_type == 'g' else ''
        
        print(f"\nAbbreviation {i+1}/{len(abbreviations)}:")
        print(f"  Text: {abbr.abbr_text} (<{element_type}{' ref=\"' + ref + '\"' if ref else ''}> element)")
        print(f"  Normalized for dictionary lookup: {abbr.normalized_form}")
        print(f"  Location (XPath): {abbr.xpath}")
        
        # Only print context before if it's not too long, otherwise abbreviate
        max_context_display = 150  # Maximum characters to display for context
        context_before = abbr.context_before
        if len(context_before) > max_context_display:
            context_before = context_before[:max_context_display] + "..."
        
        context_after = abbr.context_after
        if len(context_after) > max_context_display:
            context_after = context_after[:max_context_display] + "..."
            
        print(f"  Context: {context_before}{abbr.abbr_text}{context_after}")
        
    # Test expansion - for the first abbreviation
    if abbreviations:
        test_abbr = abbreviations[0]
        expansion = "incoming"
        print(f"\nTesting expansion '{expansion}' for '{test_abbr.abbr_text}'")
        success = processor.add_expansion(test_abbr, expansion)
        print(f"Expansion result: {'Success' if success else 'Failed'}")
        
        # Save the modified document to demonstrate the effect
        output_path = Path(__file__).parent / 'output' / 'test_expanded.xml'
        processor.save_document(tree, output_path)
        print(f"Saved expanded document to {output_path}")

if __name__ == "__main__":
    main()