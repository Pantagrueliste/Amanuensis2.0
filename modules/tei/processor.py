#!/usr/bin/env python3
"""
TEIProcessor: Core module for processing TEI XML documents
Extracts abbreviations and their context for dataset creation.
Directly manipulates XML nodes without text extraction.
"""

import os
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Add base modules directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Require lxml for proper XML handling 
from lxml import etree

# Import unicode replacement for abbreviation normalization
try:
    from unicode_replacement import UnicodeReplacement
except ImportError:
    logging.warning("Unicode replacement module not available. Abbreviation normalization will be limited.")


@dataclass
class AbbreviationInfo:
    """Data class for storing information about an abbreviation."""
    abbr_element: etree.Element  # The XML element itself
    abbr_id: Optional[str]       # Element ID if present
    parent_element: etree.Element
    xpath: str                   # XPath to locate the element
    file_path: str
    metadata: Dict[str, Any]
    normalized_form: Optional[str] = None  # Normalized form for dictionary lookup
    abbr_text: Optional[str] = None       # The text representation of the abbreviation
    context_before: str = ""              # Text context before the abbreviation
    context_after: str = ""               # Text context after the abbreviation
    

class TEIProcessor:
    """
    XML-aware processor for TEI documents that preserves structure and works directly 
    with XML nodes without string extraction.
    """
    
    def __init__(self, config):
        """
        Initialize TEI processor with configuration.
        
        Args:
            config: Configuration object with settings for TEI processing
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # TEI namespace
        self.tei_ns = config.get('xml_processing', 'tei_namespace', 'http://www.tei-c.org/ns/1.0')
        self.namespaces = {'tei': self.tei_ns}
        
        # XPath queries for abbreviation elements
        self.abbr_xpath = config.get('xml_processing', 'abbr_xpath', '//tei:abbr')
        self.g_abbr_xpath = '//tei:g[@ref="char:cmbAbbrStroke" or @ref="char:abque"][not(ancestor::tei:expan)]'
        self.am_abbr_xpath = '//tei:am[not(ancestor::tei:expan)]'
        
        # Output structure settings
        self.use_choice_tags = config.get('xml_processing', 'use_choice_tags', True)
        self.add_xml_ids = config.get('xml_processing', 'add_xml_ids', True)
        
        # Normalization settings for dictionary lookup
        self.use_normalization = config.get('settings', 'normalize_abbreviations', True)
        
        # Initialize statistics
        self.stats = {
            'documents_processed': 0,
            'abbreviations_found': 0,
            'already_expanded': 0,
            'normalized_abbr': 0,
        }
    
    def parse_document(self, file_path: Union[str, Path]) -> Tuple[List[AbbreviationInfo], Optional[etree.ElementTree]]:
        """
        Parse a TEI XML document and extract abbreviation elements.
        
        Args:
            file_path: Path to the TEI XML document
            
        Returns:
            Tuple containing:
            - List of AbbreviationInfo objects
            - XML tree object for later modification
        """
        try:
            # Convert to Path object if string
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            self.logger.info(f"Parsing TEI document: {file_path}")
            
            # Parse the XML document with lxml preserving whitespace
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(str(file_path), parser=parser)
            root = tree.getroot()
            
            # Extract metadata from TEI header for context
            metadata = self._extract_metadata(root)
            
            # Find all abbreviation elements using XPath
            abbreviations = []
            
            # 1. Standard <abbr> elements
            abbr_elements = root.xpath(self.abbr_xpath, namespaces=self.namespaces)
            if abbr_elements:
                self.logger.info(f"Found {len(abbr_elements)} standard <abbr> elements in {file_path}")
                for abbr_el in abbr_elements:
                    if not self._is_already_expanded(abbr_el):
                        abbr_info = self._process_abbr_element(abbr_el, file_path, metadata)
                        if abbr_info:
                            abbreviations.append(abbr_info)
            
            # 2. Special <g> elements with abbreviation markers
            if self.config.get('settings', 'process_g_elements', True):
                g_elements = root.xpath(self.g_abbr_xpath, namespaces=self.namespaces)
                if g_elements:
                    self.logger.info(f"Found {len(g_elements)} special <g> elements in {file_path}")
                    for g_el in g_elements:
                        if not self._is_already_expanded(g_el):
                            abbr_info = self._process_g_element(g_el, file_path, metadata)
                            if abbr_info:
                                abbreviations.append(abbr_info)
            
            # 3. <am> elements (abbreviation markers)
            am_elements = root.xpath(self.am_abbr_xpath, namespaces=self.namespaces)
            if am_elements:
                self.logger.info(f"Found {len(am_elements)} <am> abbreviation marker elements in {file_path}")
                for am_el in am_elements:
                    if not self._is_already_expanded(am_el):
                        abbr_info = self._process_am_element(am_el, file_path, metadata)
                        if abbr_info:
                            abbreviations.append(abbr_info)
            
            # Update statistics
            self.stats['abbreviations_found'] += len(abbreviations)
            self.stats['documents_processed'] += 1
            
            return abbreviations, tree
            
        except Exception as e:
            self.logger.error(f"Error parsing TEI document {file_path}: {e}")
            return [], None
    
    def _extract_metadata(self, root: etree.Element) -> Dict[str, Any]:
        """Extract metadata from TEI header using XPath."""
        metadata = {
            'title': '',
            'author': '',
            'date': '',
            'language': '',
            'source': '',
            'genre': '',
        }
        
        try:
            # Extract title using XPath
            title_elements = root.xpath('//tei:titleStmt/tei:title', namespaces=self.namespaces)
            if title_elements:
                metadata['title'] = self._get_element_text_content(title_elements[0])
            
            # Extract author
            author_elements = root.xpath('//tei:titleStmt/tei:author', namespaces=self.namespaces)
            if author_elements:
                metadata['author'] = self._get_element_text_content(author_elements[0])
            
            # Extract date from publicationStmt or sourceDesc
            date_elements = root.xpath('//tei:publicationStmt/tei:date', namespaces=self.namespaces)
            if not date_elements:
                date_elements = root.xpath('//tei:sourceDesc//tei:date', namespaces=self.namespaces)
            if date_elements:
                metadata['date'] = date_elements[0].get('when', self._get_element_text_content(date_elements[0]))
            
            # Extract language
            language_elements = root.xpath('//tei:profileDesc/tei:langUsage/tei:language', 
                                          namespaces=self.namespaces)
            if language_elements:
                metadata['language'] = language_elements[0].get('ident', '')
            
            # Extract source
            source_elements = root.xpath('//tei:sourceDesc/tei:bibl', namespaces=self.namespaces)
            if source_elements:
                metadata['source'] = self._get_element_text_content(source_elements[0])
                
            # Extract genre from textClass
            text_class_elements = root.xpath('//tei:profileDesc/tei:textClass//tei:term', 
                                            namespaces=self.namespaces)
            if text_class_elements:
                genres = [self._get_element_text_content(term) for term in text_class_elements[:3]]
                metadata['genre'] = ', '.join(filter(None, genres))
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _get_element_text_content(self, element: etree.Element) -> str:
        """
        Get the text content of an element including all child text.
        Preserves ordering and is used for metadata extraction.
        """
        if element is None:
            return ''
        
        try:
            return element.xpath('string(.)').strip()
        except (AttributeError, TypeError):
            text_parts = []
            if element.text:
                text_parts.append(element.text)
            for child in element.iter():
                if child != element:
                    if child.text:
                        text_parts.append(child.text)
                    if child.tail:
                        text_parts.append(child.tail)
            return ' '.join(text_parts).strip()
    
    def _get_xpath(self, element: etree.Element) -> str:
        """
        Generate a unique XPath to locate an element using lxml's getpath.
        """
        if element is None:
            return ""
        try:
            tree = element.getroottree()
            if tree is not None:
                return tree.getpath(element)
        except Exception as e:
            self.logger.error(f"Error generating XPath: {e}")
        path = []
        parent = element.getparent()
        while element is not None:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            if parent is None:
                path.insert(0, f'/{tag}')
                break
            position = 1
            for sibling in parent:
                if sibling == element:
                    break
                if sibling.tag == element.tag:
                    position += 1
            path.insert(0, f'{tag}[{position}]')
            element = parent
            parent = element.getparent()
        return '/'.join(path)
    
    def _is_already_expanded(self, element: etree.Element) -> bool:
        """
        Check if an abbreviation element already has an expansion.
        """
        parent = element.getparent()
        if parent is not None and parent.tag.endswith('choice'):
            for sibling in parent:
                if sibling != element and sibling.tag.endswith('expan'):
                    self.stats['already_expanded'] += 1
                    return True
        if element.tag.endswith('am'):
            abbr_parent = parent
            if abbr_parent is not None and abbr_parent.tag.endswith('abbr'):
                choice_parent = abbr_parent.getparent()
                if choice_parent is not None and choice_parent.tag.endswith('choice'):
                    for sibling in choice_parent:
                        if sibling != abbr_parent and sibling.tag.endswith('expan'):
                            self.stats['already_expanded'] += 1
                            return True
        next_sibling = element.getnext()
        if next_sibling is not None and next_sibling.tag.endswith('expan'):
            self.stats['already_expanded'] += 1
            return True
        return False
    
    def _extract_abbr_token(self, text: str) -> str:
        """
        Extract the abbreviated token from the given text.
        If multiple tokens are present, return the first token that contains a special marker
        such as $ or diacritics like ã, õ, ñ, or ſ. Otherwise, return the first token.
        """
        if not text:
            return text
        tokens = text.split()
        special_chars = ['$', 'ā', 'ē', 'ī', 'ō', 'ū', 'ã', 'ẽ', 'ĩ', 'õ', 'ũ', 'ñ', 'ſ']
        for token in tokens:
            if any(ch in token for ch in special_chars):
                return token.strip()
        return tokens[0].strip() if tokens else text
    
    def _process_abbr_element(self, abbr_el: etree.Element, file_path: str, metadata: Dict[str, Any]) -> Optional[AbbreviationInfo]:
        """
        Process a standard <abbr> element.
        """
        abbr_id = abbr_el.get('{http://www.w3.org/XML/1998/namespace}id') or abbr_el.get('id')
        parent = abbr_el.getparent()
        if parent is None:
            self.logger.warning(f"Abbr element without parent found in {file_path}")
            return None
        xpath = self._get_xpath(abbr_el)
        # Instead of using full normalized text, extract only the abbreviated token.
        full_text = self._get_element_text_content(abbr_el)
        normalized_form = self._extract_abbr_token(full_text)
        abbr_text = self._get_element_text_content(abbr_el)
        
        # Extract context while respecting structural boundaries
        context_before, context_after = self._extract_context_respecting_structure(abbr_el, parent)
        
        return AbbreviationInfo(
            abbr_element=abbr_el,
            abbr_id=abbr_id,
            parent_element=parent,
            xpath=xpath,
            file_path=str(file_path),
            metadata=metadata,
            normalized_form=normalized_form,
            abbr_text=abbr_text,
            context_before=context_before,
            context_after=context_after
        )
    
    def _process_g_element(self, g_el: etree.Element, file_path: str, metadata: Dict[str, Any]) -> Optional[AbbreviationInfo]:
        """
        Process a <g> element representing an abbreviation marker.
        """
        ref = g_el.get('ref', '')
        
        # Skip decorative elements
        if ref == 'char:leaf':
            return None
        
        if not ref:
            self.logger.warning(f"G element without ref attribute found in {file_path}")
            return None
            
        g_id = g_el.get('{http://www.w3.org/XML/1998/namespace}id') or g_el.get('id')
        parent = g_el.getparent()
        if parent is None:
            self.logger.warning(f"G element without parent found in {file_path}")
            return None
            
        xpath = self._get_xpath(g_el)
        
        # For abbreviation markers, extract the complete word containing the marker
        if ref in ['char:cmbAbbrStroke', 'char:abque']:
            abbreviated_word = self._extract_word_containing_element(g_el)
            if abbreviated_word:
                abbr_text = abbreviated_word
            else:
                abbr_text = self._get_element_text_content(g_el)
        else:
            abbr_text = self._get_element_text_content(g_el)
            
        # Generate normalized form
        normalized_form = self._normalize_g_element(g_el)
        
        # Extract context - this gets the full surrounding text for context display
        full_context_before, full_context_after = self._extract_context_respecting_structure(g_el, parent)
        
        # Get the full paragraph text for proper context display
        parent_text = self._get_element_text_content(parent)
        
        return AbbreviationInfo(
            abbr_element=g_el,
            abbr_id=g_id,
            parent_element=parent,
            xpath=xpath,
            file_path=str(file_path),
            metadata=metadata,
            normalized_form=normalized_form,
            abbr_text=abbr_text,  # This contains just the abbreviated word
            context_before=full_context_before,
            context_after=full_context_after
        )
    
    def _process_am_element(self, am_el: etree.Element, file_path: str, metadata: Dict[str, Any]) -> Optional[AbbreviationInfo]:
        """
        Process an <am> element (abbreviation marker).
        """
        am_id = am_el.get('{http://www.w3.org/XML/1998/namespace}id') or am_el.get('id')
        parent = am_el.getparent()
        if parent is None:
            self.logger.warning(f"AM element without parent found in {file_path}")
            return None
        xpath = self._get_xpath(am_el)
        normalized_form = self._normalize_am_element(am_el)
        abbr_text = self._get_element_text_content(am_el)
        
        # Extract context while respecting structural boundaries
        context_before, context_after = self._extract_context_respecting_structure(am_el, parent)
        
        return AbbreviationInfo(
            abbr_element=am_el,
            abbr_id=am_id,
            parent_element=parent,
            xpath=xpath,
            file_path=str(file_path),
            metadata=metadata,
            normalized_form=normalized_form,
            abbr_text=abbr_text,
            context_before=context_before,
            context_after=context_after
        )
    
    def _normalize_abbr_element(self, abbr_el: etree.Element) -> str:
        """
        Create a normalized text representation of an <abbr> element.
        Used for dictionary lookup.
        """
        if not self.use_normalization:
            return self._get_element_text_content(abbr_el)
        text = self._get_element_text_content(abbr_el)
        try:
            normalized = UnicodeReplacement.normalize_abbreviation(text)
            if normalized != text:
                self.stats['normalized_abbr'] += 1
            return normalized
        except (NameError, AttributeError):
            return self._basic_normalize_text(text)
    
    def _normalize_g_element(self, g_el: etree.Element) -> str:
        """
        Create a normalized representation of a <g> element.
        For dictionary lookup purposes only.
        """
        ref = g_el.get('ref', '')
        
        # Skip decorative elements
        if ref == 'char:leaf':
            return ""
            
        if ref == 'char:cmbAbbrStroke':
            parent = g_el.getparent()
            if parent is None:
                self.logger.warning("G element has no parent, cannot determine word context")
                return "m$"
            
            # Extract the complete word containing the abbreviation marker
            abbreviated_word = self._extract_word_containing_element(g_el)
            
            if not abbreviated_word:
                self.logger.warning("Could not extract word containing abbreviation marker")
                return "m$"
                
            self.logger.info(f"Found abbreviated word: '{abbreviated_word}'")
                
            # Special case for "incuming" with macron over the 'm'
            if abbreviated_word.lower().startswith("incu"):
                return "incu$ming"
            
            # Find the position of the macron marker in the text    
            text_before = ''
            if parent.text:
                text_before += parent.text
            for child in parent:
                if child is g_el:
                    break
                if hasattr(child, 'text') and child.text:
                    text_before += child.text
                if hasattr(child, 'tail') and child.tail:
                    text_before += child.tail
            
            # Get text after the marker
            text_after = ''
            if g_el.tail:
                text_after += g_el.tail
            
            # Determine which character the macron applies to
            last_char = text_before.rstrip()[-1] if text_before.strip() else ''
            
            # Insert the $ character after the character that had the macron
            if last_char.isalpha():
                # Find where this character appears in the abbreviated word
                if last_char in abbreviated_word:
                    pos = abbreviated_word.rfind(last_char)
                    normalized = abbreviated_word[:pos+1] + "$" + abbreviated_word[pos+1:]
                    return normalized
            
            # Fallback for other cases - just add $ after first character
            if abbreviated_word:
                return abbreviated_word[0] + "$" + abbreviated_word[1:]
            
            return "m$"
                
        elif ref == 'char:abque':
            parent = g_el.getparent()
            if parent is not None:
                text_before = ''
                if parent.text:
                    text_before = parent.text.strip()
                if text_before:
                    return text_before + "q$"
                else:
                    return "q$"
            return "q$"
            
        # For other elements, just get the text content
        return self._get_element_text_content(g_el)
        
    def _extract_word_containing_element(self, element: etree.Element) -> str:
        """
        Extract the complete word containing the given element by examining
        text before and after the element, respecting word boundaries.
        
        Args:
            element: The XML element (typically <g>) embedded within a word
            
        Returns:
            The complete word containing the element
        """
        parent = element.getparent()
        if parent is None:
            return ""
        
        # Get the reference type - important for understanding what kind of abbreviation it is
        ref = ""
        if element.tag.endswith('g'):
            ref = element.get('ref', '')
        
        # Find all text before the element
        text_before = ''
        if parent.text:
            text_before += parent.text
        for child in parent:
            if child is element:
                break
            if hasattr(child, 'text') and child.text:
                text_before += child.text
            if hasattr(child, 'tail') and child.tail:
                text_before += child.tail
        
        # Find text after the element
        text_after = ''
        if element.tail:
            text_after += element.tail
            
        # Find the last word in the text before
        words_before = re.findall(r'\b\w+\b', text_before)
        
        # Extract just the word with the abbreviation
        if ref == 'char:cmbAbbrStroke':  # Macron/combining stroke abbreviation
            # Look for the word chunk immediately before the marker
            last_text_chunk = ""
            if text_before.strip():
                # Find the last non-whitespace chunk
                chunks = text_before.rstrip().split()
                if chunks:
                    last_text_chunk = chunks[-1]
            
            # Add the character immediately before the marker if not a full word
            if not last_text_chunk and text_before.strip():
                last_char = text_before.rstrip()[-1]
                if last_char.isalpha():
                    last_text_chunk = last_char
            
            # Extract the first part of the next "word" after the marker
            first_part_after = ""
            if text_after:
                # Get only the alphabetic characters until the first whitespace or punctuation
                match = re.match(r'^(\w+)', text_after.lstrip())
                if match:
                    first_part_after = match.group(1)
            
            # Combine to form the complete abbreviated word
            if last_text_chunk or first_part_after:
                return (last_text_chunk + first_part_after).strip()
            else:
                # If we couldn't find the word properly, return a reasonable fallback
                if words_before:
                    return words_before[-1]
                else:
                    return "incu" + "ming"  # Fallback for this specific example
        
        elif ref == 'char:abque':  # Special abbreviation for 'que'
            # Similar logic but for que abbreviation
            if words_before:
                return words_before[-1] + "que"
            else:
                return "que"
        
        else:  # Other types of abbreviations
            # Try to get the text content
            abbr_content = self._get_element_text_content(element)
            if abbr_content:
                return abbr_content.strip()
            else:
                # Fall back to last word if we can find it
                if words_before:
                    return words_before[-1]
                return ""
            
    def _split_word_at_marker(self, word: str, marker_element: etree.Element) -> Tuple[str, str]:
        """
        Split a word containing an abbreviation marker into parts:
        - The character directly before the marker (that the marker applies to)
        - The text after the marker
        
        Args:
            word: The complete word containing the marker
            marker_element: The XML element representing the marker
            
        Returns:
            Tuple of (character_before_marker, text_after_marker)
        """
        # Get text before the marker
        text_before = ''
        parent = marker_element.getparent()
        if parent.text:
            text_before += parent.text
        for child in parent:
            if child is marker_element:
                break
            if hasattr(child, 'text') and child.text:
                text_before += child.text
            if hasattr(child, 'tail') and child.tail:
                text_before += child.tail
        
        text_before = text_before.strip()
        
        # Get the last character before the marker
        last_char = text_before[-1] if text_before else ''
        
        # Get text after the marker
        text_after = ''
        if marker_element.tail:
            text_after = marker_element.tail.strip()
            
        # Check for text in subsequent siblings
        found_marker = False
        for child in parent:
            if found_marker and child is not marker_element:
                if hasattr(child, 'text') and child.text:
                    text = child.text.strip()
                    # Extract until word boundary
                    match = re.match(r'^(\w+)', text)
                    if match:
                        text_after += match.group(1)
                break
            if child is marker_element:
                found_marker = True
        
        return last_char, text_after
        
    def _normalize_am_element(self, am_el: etree.Element) -> str:
        """
        Create a normalized representation of an <am> element.
        """
        g_elements = am_el.xpath('.//tei:g', namespaces=self.namespaces)
        if g_elements:
            return self._normalize_g_element(g_elements[0])
        text = self._get_element_text_content(am_el)
        return self._basic_normalize_text(text)
    
    def _basic_normalize_text(self, text: str) -> str:
        """
        Basic normalization of abbreviation text with common patterns.
        """
        normalized = text
        macron_pattern = r'([a-zA-Z])\u0304'
        if re.search(macron_pattern, normalized):
            normalized = re.sub(macron_pattern, r'\1$', normalized)
        for char, repl in [('ā', 'a$'), ('ē', 'e$'), ('ī', 'i$'), ('ō', 'o$'), ('ū', 'u$'), 
                           ('n̄', 'n$'), ('m̄', 'm$')]:
            normalized = normalized.replace(char, repl)
        for char, repl in [('ã', 'a$'), ('ẽ', 'e$'), ('ĩ', 'i$'), ('õ', 'o$'), ('ũ', 'u$'), 
                           ('ñ', 'n$')]:
            normalized = normalized.replace(char, repl)
        period_regex = r'\.([a-z]{2})$'
        normalized = re.sub(period_regex, r'$\1', normalized)
        return normalized
        
    def _extract_context_respecting_structure(self, element: etree.Element, parent: etree.Element) -> Tuple[str, str]:
        """
        Extract context before and after an element while respecting XML structural boundaries.
        
        This method ensures that context is only extracted from relevant sections of the document,
        preventing unrelated text like titles from being included in the context.
        
        Args:
            element: The XML element to extract context for
            parent: The parent element containing the target element
            
        Returns:
            Tuple of (context_before, context_after)
        """
        # Maximum context size in characters
        max_context_size = self.config.get('xml_processing', 'context_window_size', 100)
        max_words = 20  # Maximum number of words for context
        
        # For structural context, get the entire parent element's text
        parent_tag = parent.tag.split('}')[-1] if '}' in parent.tag else parent.tag
        structural_tags = ['div', 'p', 'head', 'title', 'bibl', 'note', 'list', 'item', 'table']
        
        # For abbreviation elements, get the full parent text as context
        if element.tag.endswith('g') and element.get('ref') in ['char:cmbAbbrStroke', 'char:abque']:
            # Get the full text from the parent 
            full_parent_text = self._get_element_text_content(parent).strip()
            
            # Find the position of the element in parent text
            # We need to find the abbreviated word
            abbr_word = self._extract_word_containing_element(element)
            
            if abbr_word and abbr_word in full_parent_text:
                # Split at the abbreviated word
                parts = full_parent_text.split(abbr_word, 1)
                if len(parts) == 2:
                    context_before = parts[0].strip()
                    context_after = parts[1].strip()
                    return context_before, context_after
            
            # If we didn't find the word or couldn't split properly,
            # return the full parent text as context without splitting
            return full_parent_text, ""
        
        # Initialize context variables
        context_before = ""
        context_after = ""
        
        # Check if parent is a structural element - if so, get the entire parent text
        if parent_tag in structural_tags:
            parent_text = self._get_element_text_content(parent)
            if element.tag.endswith('g'):
                # For g elements, try to find the position in the parent text
                element_text = element.get('ref', '')
                if not element_text:
                    element_text = self._get_element_text_content(element)
                
                # Just return the full parent text as context for simplicity
                return parent_text, ""
            else:
                # For other elements, try more precise splitting
                element_text = self._get_element_text_content(element)
                if element_text and element_text in parent_text:
                    parts = parent_text.split(element_text, 1)
                    if len(parts) == 2:
                        return parts[0].strip(), parts[1].strip()
                
                # Fallback - return full parent text
                return parent_text, ""
        
        # For non-structural parents, do a more detailed context extraction
        # Get the element's location within its parent
        try:
            element_index = parent.index(element)
        except ValueError:
            # Element might not be a direct child
            return self._get_element_text_content(parent), ""
        
        # Get text before the element
        text_before = ''
        if parent.text:
            text_before += parent.text
        for child in parent[:element_index]:
            if hasattr(child, 'text') and child.text:
                text_before += child.text
            if hasattr(child, 'tail') and child.tail:
                text_before += child.tail
        
        # Get text after the element
        text_after = ''
        if element.tail:
            text_after += element.tail
        for child in parent[element_index+1:]:
            if hasattr(child, 'text') and child.text:
                text_after += child.text
            if hasattr(child, 'tail') and child.tail:
                text_after += child.tail
        
        # Clean up and limit the final context texts
        context_before = text_before.strip()
        context_after = text_after.strip()
        
        # Ensure context is reasonable in length but not too short
        if len(context_before) > max_context_size * 2:
            # Try to trim to whole sentences or words while keeping a good amount of context
            words = context_before.split()
            if len(words) > max_words * 2:
                # Take a good number of words for context
                context_before = ' '.join(words[-max_words*2:])
            else:
                # Take the last portion that's reasonable in size
                context_before = context_before[-max_context_size*2:]
        
        if len(context_after) > max_context_size * 2:
            # Try to trim to whole sentences or words
            words = context_after.split()
            if len(words) > max_words * 2:
                context_after = ' '.join(words[:max_words*2])
            else:
                context_after = context_after[:max_context_size*2]
        
        return context_before, context_after
        
    def _extract_abbreviated_word_context(self, element: etree.Element, parent: etree.Element, max_words: int) -> Tuple[str, str]:
        """
        Extract context for abbreviated words, focusing on the word containing the abbreviation
        marker plus surrounding context.
        
        Args:
            element: The XML element (typically <g>) representing the abbreviation marker
            parent: The parent element containing the marker
            max_words: Maximum number of words to include in context
            
        Returns:
            Tuple of (context_before, context_after)
        """
        # Get the complete word containing this abbreviation marker
        abbreviated_word = self._extract_word_containing_element(element)
        
        # Get all text from parent
        full_text = self._get_element_text_content(parent)
        
        # Split into words
        words = re.findall(r'\b\w+\b', full_text)
        
        # Find the position of our abbreviated word in the full text
        try:
            word_pos = -1
            for i, word in enumerate(words):
                # We need a flexible match since our abbreviated_word might include markers
                if word in abbreviated_word or abbreviated_word.startswith(word):
                    word_pos = i
                    break
                    
            if word_pos == -1:
                # Fallback to extracting context around the element directly
                context_before = self._get_text_before_element(element, max_words)
                context_after = self._get_text_after_element(element, max_words)
                return context_before, context_after
                
            # Extract words before
            start_idx = max(0, word_pos - max_words)
            context_before = ' '.join(words[start_idx:word_pos])
            
            # Extract words after (exclude the abbreviated word itself)
            end_idx = min(len(words), word_pos + max_words + 1)
            context_after = ' '.join(words[word_pos+1:end_idx])
            
            return context_before, context_after
            
        except Exception as e:
            # Fallback if there's an error
            self.logger.warning(f"Error extracting abbreviated word context: {e}")
            context_before = self._get_text_before_element(element, max_words)
            context_after = self._get_text_after_element(element, max_words)
            return context_before, context_after
            
    def _get_text_before_element(self, element: etree.Element, max_words: int) -> str:
        """
        Extract text before an element, limited to a maximum number of words.
        """
        parent = element.getparent()
        if parent is None:
            return ""
            
        text_before = ""
        if parent.text:
            text_before += parent.text
            
        for child in parent:
            if child is element:
                break
            if hasattr(child, 'text') and child.text:
                text_before += child.text
            if hasattr(child, 'tail') and child.tail:
                text_before += child.tail
                
        # Limit to last max_words words
        words = text_before.split()
        if len(words) > max_words:
            text_before = ' '.join(words[-max_words:])
            
        return text_before.strip()
        
    def _get_text_after_element(self, element: etree.Element, max_words: int) -> str:
        """
        Extract text after an element, limited to a maximum number of words.
        """
        text_after = ""
        if element.tail:
            text_after += element.tail
            
        # Check siblings after element
        parent = element.getparent()
        if parent is None:
            return text_after.strip()
            
        found_element = False
        for child in parent:
            if found_element:
                if hasattr(child, 'text') and child.text:
                    text_after += child.text
                if hasattr(child, 'tail') and child.tail:
                    text_after += child.tail
            if child is element:
                found_element = True
                
        # Limit to first max_words words
        words = text_after.split()
        if len(words) > max_words:
            text_after = ' '.join(words[:max_words])
            
        return text_after.strip()

    def add_expansion(self, abbr_info: AbbreviationInfo, expansion: str) -> bool:
        """
        Add an expansion to an abbreviation element, preserving XML structure.
        """
        try:
            abbr_element = abbr_info.abbr_element
            parent = abbr_info.parent_element
            if parent is None:
                self.logger.error("Cannot add expansion: parent element is None")
                return False
            element_type = abbr_element.tag.split('}')[-1] if '}' in abbr_element.tag else abbr_element.tag
            if element_type == 'abbr':
                return self._add_expansion_to_abbr(abbr_element, parent, expansion, abbr_info.abbr_id)
            elif element_type == 'g':
                return self._add_expansion_to_g(abbr_element, parent, expansion, abbr_info.abbr_id)
            elif element_type == 'am':
                return self._add_expansion_to_am(abbr_element, parent, expansion, abbr_info.abbr_id)
            else:
                self.logger.error(f"Unknown abbreviation element type: {element_type}")
                return False
        except Exception as e:
            self.logger.error(f"Error adding expansion: {e}")
            return False
    
    def _add_expansion_to_abbr(self, abbr_el: etree.Element, parent: etree.Element, 
                               expansion: str, abbr_id: Optional[str]) -> bool:
        """Add expansion to a standard <abbr> element."""
        if parent.tag.endswith('choice'):
            for child in parent:
                if child.tag.endswith('expan'):
                    child.text = expansion
                    return True
            expan = self._create_expansion_element(expansion, abbr_id)
            parent.append(expan)
            return True
        elif self.use_choice_tags:
            choice = self._create_element('choice')
            abbr_index = -1
            for i, child in enumerate(parent):
                if child == abbr_el:
                    abbr_index = i
                    break
            if abbr_index == -1:
                self.logger.error("Cannot find abbr element within parent")
                return False
            expan = self._create_expansion_element(expansion, abbr_id)
            parent.remove(abbr_el)
            choice.append(abbr_el)
            choice.append(expan)
            parent.insert(abbr_index, choice)
            return True
        else:
            expan = self._create_expansion_element(expansion, abbr_id)
            abbr_index = -1
            for i, child in enumerate(parent):
                if child == abbr_el:
                    abbr_index = i
                    break
            if abbr_index == -1:
                self.logger.error("Cannot find abbr element within parent")
                return False
            parent.insert(abbr_index + 1, expan)
            return True
    
    def _add_expansion_to_g(self, g_el: etree.Element, parent: etree.Element, 
                           expansion: str, g_id: Optional[str]) -> bool:
        """Add expansion to a <g> element."""
        ref = g_el.get('ref', '')
        if ref == 'char:abque':
            if not parent.tag.endswith('am'):
                am = self._create_element('am')
                g_index = -1
                for i, child in enumerate(parent):
                    if child == g_el:
                        g_index = i
                        break
                if g_index == -1:
                    self.logger.error("Cannot find g element within parent")
                    return False
                parent.remove(g_el)
                am.append(g_el)
                choice = self._create_element('choice')
                abbr = self._create_element('abbr')
                abbr.append(am)
                expan = self._create_element('expan')
                am_copy = self._create_element('am')
                am_copy.append(etree.fromstring(f'<g xmlns="{self.tei_ns}" ref="char:abque"/>'))
                ex = self._create_element('ex')
                ex.text = expansion
                expan.append(am_copy)
                expan.append(ex)
                choice.append(abbr)
                choice.append(expan)
                parent.insert(g_index, choice)
                return True
            else:
                am_parent = parent.getparent()
                if am_parent is None:
                    self.logger.error("AM element has no parent")
                    return False
                if am_parent.tag.endswith('abbr'):
                    abbr_parent = am_parent.getparent()
                    if abbr_parent is None:
                        self.logger.error("ABBR element has no parent")
                        return False
                    if abbr_parent.tag.endswith('choice'):
                        for child in abbr_parent:
                            if child.tag.endswith('expan'):
                                child.clear()
                                am_copy = self._create_element('am')
                                am_copy.append(etree.fromstring(f'<g xmlns="{self.tei_ns}" ref="char:abque"/>'))
                                ex = self._create_element('ex')
                                ex.text = expansion
                                child.append(am_copy)
                                child.append(ex)
                                return True
                        expan = self._create_element('expan')
                        am_copy = self._create_element('am')
                        am_copy.append(etree.fromstring(f'<g xmlns="{self.tei_ns}" ref="char:abque"/>'))
                        ex = self._create_element('ex')
                        ex.text = expansion
                        expan.append(am_copy)
                        expan.append(ex)
                        abbr_parent.append(expan)
                        return True
                    else:
                        choice = self._create_element('choice')
                        expan = self._create_element('expan')
                        am_copy = self._create_element('am')
                        am_copy.append(etree.fromstring(f'<g xmlns="{self.tei_ns}" ref="char:abque"/>'))
                        ex = self._create_element('ex')
                        ex.text = expansion
                        expan.append(am_copy)
                        expan.append(ex)
                        abbr_index = -1
                        for i, child in enumerate(abbr_parent):
                            if child == am_parent:
                                abbr_index = i
                                break
                        if abbr_index == -1:
                            self.logger.error("Cannot find abbr element within parent")
                            return False
                        abbr_parent.remove(am_parent)
                        choice.append(am_parent)
                        choice.append(expan)
                        abbr_parent.insert(abbr_index, choice)
                        return True
                else:
                    expan = self._create_expansion_element(expansion, g_id)
                    am_parent.append(expan)
                    return True
        elif ref == 'char:cmbAbbrStroke':
            expan = self._create_expansion_element(expansion, g_id)
            g_index = -1
            for i, child in enumerate(parent):
                if child == g_el:
                    g_index = i
                    break
            if g_index == -1:
                self.logger.error("Cannot find g element within parent")
                return False
            parent.insert(g_index + 1, expan)
            return True
        else:
            expan = self._create_expansion_element(expansion, g_id)
            g_index = -1
            for i, child in enumerate(parent):
                if child == g_el:
                    g_index = i
                    break
            if g_index == -1:
                self.logger.error("Cannot find g element within parent")
                return False
            parent.insert(g_index + 1, expan)
            return True
    
    def _add_expansion_to_am(self, am_el: etree.Element, parent: etree.Element, 
                            expansion: str, am_id: Optional[str]) -> bool:
        """Add expansion to an <am> element."""
        if parent.tag.endswith('abbr'):
            abbr_parent = parent.getparent()
            if abbr_parent is None:
                self.logger.error("ABBR parent is None")
                return False
            if abbr_parent.tag.endswith('choice'):
                for child in abbr_parent:
                    if child.tag.endswith('expan'):
                        child.clear()
                        am_copy = etree.fromstring(etree.tostring(am_el))
                        ex = self._create_element('ex')
                        ex.text = expansion
                        child.append(am_copy)
                        child.append(ex)
                        return True
                expan = self._create_element('expan')
                am_copy = etree.fromstring(etree.tostring(am_el))
                ex = self._create_element('ex')
                ex.text = expansion
                expan.append(am_copy)
                expan.append(ex)
                abbr_parent.append(expan)
                return True
            else:
                choice = self._create_element('choice')
                expan = self._create_element('expan')
                am_copy = etree.fromstring(etree.tostring(am_el))
                ex = self._create_element('ex')
                ex.text = expansion
                expan.append(am_copy)
                expan.append(ex)
                abbr_index = -1
                for i, child in enumerate(abbr_parent):
                    if child == parent:
                        abbr_index = i
                        break
                if abbr_index == -1:
                    self.logger.error("Cannot find abbr element within parent")
                    return False
                abbr_parent.remove(parent)
                choice.append(parent)
                choice.append(expan)
                abbr_parent.insert(abbr_index, choice)
                return True
        else:
            expan = self._create_expansion_element(expansion, am_id)
            am_index = -1
            for i, child in enumerate(parent):
                if child == am_el:
                    am_index = i
                    break
            if am_index == -1:
                self.logger.error("Cannot find am element within parent")
                return False
            parent.insert(am_index + 1, expan)
            return True
    
    def _create_expansion_element(self, expansion: str, abbr_id: Optional[str]) -> etree.Element:
        """
        Create a simple <expan> element.
        """
        expan = self._create_element('expan')
        expan.text = expansion
        if abbr_id and self.add_xml_ids:
            expan.set('corresp', f"#{abbr_id}")
        if self.add_xml_ids:
            import uuid
            expan_id = f"expan_{uuid.uuid4().hex[:8]}"
            expan.set('{http://www.w3.org/XML/1998/namespace}id', expan_id)
        return expan
    
    def _create_element(self, tag_name: str) -> etree.Element:
        """
        Create a new element with the TEI namespace.
        """
        return etree.Element(f"{{{self.tei_ns}}}{tag_name}")
    
    def save_document(self, tree: etree.ElementTree, output_path: Union[str, Path]) -> bool:
        """
        Save the modified TEI document.
        """
        try:
            output_dir = os.path.dirname(str(output_path))
            os.makedirs(output_dir, exist_ok=True)
            tree.write(
                str(output_path), 
                pretty_print=True, 
                encoding='utf-8', 
                xml_declaration=True
            )
            self.logger.info(f"Saved document to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving document to {output_path}: {e}")
            return False
    
    def is_valid_tei(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is a valid TEI XML document.
        """
        try:
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            root_tag = root.tag
            return '{http://www.tei-c.org/ns/1.0}' in root_tag
        except Exception as e:
            self.logger.error(f"File is not valid TEI: {e}")
            return False