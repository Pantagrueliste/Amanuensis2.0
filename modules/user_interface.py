#!/usr/bin/env python3
"""
User Interface for Amanuensis 2.0 - Command Line Interface

This module provides a rich command-line interface for interacting with the Amanuensis 2.0
abbreviation expansion system.
"""

import os
import sys
import argparse
import logging
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
from lxml import etree

from .config import Config
from .tei.processor import TEIProcessor, AbbreviationInfo
from .suggestion_generator import SuggestionGenerator
from .dataset.dataset_builder import DatasetBuilder

console = Console()


class UserInterface:
    """
    Command-line interface for Amanuensis 2.0 that provides interactive 
    abbreviation expansion capabilities.
    """
    
    def __init__(self, config_path: str = "config.toml"):
        """
        Initialize the user interface.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = Config(config_path)
        self.tei_processor = TEIProcessor(self.config)
        self.suggestion_generator = SuggestionGenerator(self.config)
        self.dataset_builder = DatasetBuilder(self.config)
        
        # For keeping track of user decisions
        self.user_decisions = {}
        self.abbreviations_processed = 0
        self.abbreviations_expanded = 0
        self.files_processed = 0
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, sig, frame):
        """Handle Ctrl+C by saving work and exiting gracefully."""
        console.print("\n[bold red]Interrupted! Saving current progress...[/bold red]")
        self._save_user_decisions()
        console.print("[bold green]Progress saved. Exiting Amanuensis 2.0[/bold green]")
        sys.exit(0)
    
    def _save_user_decisions(self):
        """
        Save user decisions to file and create dataset entries.
        """
        output_dir = Path(self.config.get("paths", "output_path"))
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw decisions first
        decisions_file = output_dir / f"user_decisions_{timestamp}.json"
        
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_decisions, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved raw user decisions to {decisions_file}")
        
        # Convert user decisions to dataset format
        dataset_entries = []
        for abbr_text, decision in self.user_decisions.items():
            entry = {
                'abbreviation': abbr_text,
                'expansion': decision['expansion'],
                'context_before': decision.get('context_before', ''),
                'context_after': decision.get('context_after', ''),
                'source': {
                    'file': decision.get('file_path', ''),
                    'confidence': decision.get('confidence', 1.0),
                    'source_type': decision.get('source', 'user')
                }
            }
            
            # Include metadata if available
            if 'metadata' in decision and decision['metadata']:
                entry['metadata'] = decision['metadata']
                
            dataset_entries.append(entry)
        
        # Save as structured dataset
        dataset_dir = output_dir / "datasets"
        dataset_dir.mkdir(exist_ok=True)
        dataset_file = dataset_dir / f"expansion_dataset_{timestamp}.json"
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_entries, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(dataset_entries)} dataset entries to {dataset_file}")
        console.print(f"[green]Saved user decisions to {decisions_file}[/green]")
        console.print(f"[green]Created dataset with {len(dataset_entries)} entries at {dataset_file}[/green]")
    
    def show_welcome(self):
        """Display welcome message and application info."""
        title = "AMANUENSIS 2.0"
        subtitle = "Early Modern Abbreviation Expansion System"
        
        console.print(Panel.fit(
            f"[bold yellow]{title}[/bold yellow]\n[italic]{subtitle}[/italic]",
            border_style="bright_blue"
        ))
        
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Input path: [cyan]{self.config.get('paths', 'input_path')}[/cyan]")
        console.print(f"  Output path: [cyan]{self.config.get('paths', 'output_path')}[/cyan]")
        console.print(f"  Language model: [cyan]{self.config.get('language_model_integration', 'provider')} - {self.config.get('language_model_integration', 'model_name')}[/cyan]")
        
        console.print("\n[bold]Ready to collect abbreviation expansion examples.[/bold]")
    
    def show_main_menu(self) -> str:
        """
        Display the main menu and get user choice.
        
        Returns:
            User's menu choice
        """
        console.print("\n[bold]Main Menu:[/bold]")
        options = [
            "1. Extract abbreviations from TEI documents",
            "2. Interactive abbreviation expansion",
            "3. Build training dataset from collected examples",
            "4. View statistics",
            "5. Settings",
            "6. Exit"
        ]
        
        for option in options:
            console.print(f"  {option}")
        
        # Show active format setting if present
        current_format = self.config.get('dataset', 'format', 'json')
        format_name = {
            'json': 'JSON (Standard)',
            'jsonl': 'JSONL (Line-oriented)',
            'tei_xml': 'TEI XML (Preserves structure)'
        }.get(current_format, current_format)
        
        console.print(f"\n[dim]Current dataset format: {format_name}[/dim]")
        
        try:
            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5", "6"])
            return choice
        except KeyboardInterrupt:
            console.print("[yellow]Keyboard interrupt. Exiting...[/yellow]")
            import sys
            sys.exit(0)
        except EOFError:
            console.print("[yellow]Input interrupted. Please try again.[/yellow]")
            # Return a default choice that's safe (View Statistics)
            return "4"
    
    def process_tei_documents(self):
        """
        Extract abbreviations from TEI documents and collect expansion examples.
        If interactive_mode is True, skip the continuous Rich progress bar 
        to avoid interfering with user input.
        """
        input_path = self.config.get("paths", "input_path")
        output_path = self.config.get("paths", "output_path")
        target_language = self.config.get("settings", "language", "eng")
        discarded_dir = self.config.get("paths", "discarded_directory", 
                                       os.path.join(output_path, "discarded"))
        
        # Ensure discarded directory exists
        os.makedirs(discarded_dir, exist_ok=True)
        
        # Find all XML files in the input directory
        xml_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_files.append(os.path.join(root, file))
        
        if not xml_files:
            console.print("[yellow]No XML files found in the input directory.[/yellow]")
            return
        
        console.print(f"[bold]Found {len(xml_files)} XML files to process.[/bold]")
        process_all = Confirm.ask("Process all files?")
        
        if not process_all:
            # Show file list and let user select
            table = Table(title="Available XML Files")
            table.add_column("Index", style="cyan")
            table.add_column("File Path", style="green")
            
            for idx, file_path in enumerate(xml_files, 1):
                rel_path = os.path.relpath(file_path, input_path)
                table.add_row(str(idx), rel_path)
            
            console.print(table)
            
            selected_indices = Prompt.ask(
                "Enter file numbers to process (comma-separated, e.g., 1,3,5)",
                default="1"
            )
            
            try:
                indices = [int(idx.strip()) for idx in selected_indices.split(",")]
                selected_files = [xml_files[idx-1] for idx in indices if 1 <= idx <= len(xml_files)]
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection. Using the first file.[/bold red]")
                selected_files = [xml_files[0]]
        else:
            selected_files = xml_files
        
        # Check for language mismatch
        files_to_process = []
        files_with_mismatched_language = []
        
        console.print(f"[bold]Checking document languages (target: {target_language})...[/bold]")
        for file_path in selected_files:
            try:
                # Parse the XML document just to check the language
                parser = etree.XMLParser(remove_blank_text=True)
                tree = etree.parse(str(file_path), parser=parser)
                root = tree.getroot()
                
                # Get document language
                doc_language = self.tei_processor.get_document_language(root)
                if doc_language and doc_language != target_language:
                    files_with_mismatched_language.append((file_path, doc_language))
                else:
                    files_to_process.append(file_path)
            except Exception as e:
                self.logger.error(f"Error checking language in {file_path}: {e}")
                files_to_process.append(file_path)  # Process it anyway
        
        # Handle files with mismatched language
        if files_with_mismatched_language:
            console.print(f"[yellow]Found {len(files_with_mismatched_language)} files with language different from target ({target_language}):[/yellow]")
            
            lang_table = Table(title=f"Files with Non-{target_language} Language")
            lang_table.add_column("File", style="cyan")
            lang_table.add_column("Language", style="yellow")
            
            for file_path, lang in files_with_mismatched_language:
                rel_path = os.path.relpath(file_path, input_path)
                lang_table.add_row(rel_path, lang)
            
            console.print(lang_table)
            
            if Confirm.ask("Would you like to discard these files?"):
                for file_path, _ in files_with_mismatched_language:
                    # Move to discarded directory
                    rel_path = os.path.relpath(file_path, input_path)
                    dest_path = os.path.join(discarded_dir, os.path.basename(file_path))
                    import shutil
                    shutil.copy2(file_path, dest_path)
                    console.print(f"[bold]Moved {rel_path} to discarded directory[/bold]")
            else:
                # Add them back to files to process
                files_to_process.extend([f[0] for f in files_with_mismatched_language])
        
        if not files_to_process:
            console.print("[yellow]No files to process after language filtering.[/yellow]")
            return
        
        console.print(f"[bold]Processing {len(files_to_process)} files...[/bold]")
        selected_files = files_to_process
        
        use_interactive = self.config.get("user_interface", "interactive_mode", True)
        
        if use_interactive:
            # No continuous progress bar; just process each file and show a summary line
            i = 1
            while i <= len(selected_files):
                file_path = selected_files[i-1]
                console.print(f"[bold magenta]\nProcessing file {i}/{len(selected_files)}:[/bold magenta] {file_path}")
                
                # Process the file - asking about discarding is now handled per abbreviation
                self._process_single_tei_file(file_path, output_path)
                
                i += 1
            console.print(f"[bold green]Finished interactive expansion for {len(selected_files)} files.[/bold green]")
        else:
            # Use Rich Progress bar for non-interactive mode
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing files...", total=len(selected_files))
                for file_path in selected_files:
                    rel_path = os.path.relpath(file_path, input_path)
                    progress.update(task, description=f"[cyan]Processing {rel_path}...[/cyan]")
                    self._process_single_tei_file(file_path, output_path)
                    progress.update(task, advance=1)
        
        console.print(f"[bold green]Processed {self.files_processed} files, "
                      f"collected {self.abbreviations_expanded} abbreviation expansions.[/bold green]")
    
    def _process_single_tei_file(self, file_path: str, output_dir: str):
        """
        Process a single TEI XML file by extracting abbreviations, 
        then performing interactive or automatic expansions.
        """
        try:
            # Track file-specific decisions separately
            file_decisions = {}
            discard_file = False
            
            # Extract abbreviations
            abbreviations, tree = self.tei_processor.parse_document(file_path)
            
            if not abbreviations:
                self.logger.info(f"No abbreviations found in {file_path}")
                return
            
            # Create output directory structure
            rel_path = os.path.relpath(file_path, self.config.get("paths", "input_path"))
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Decide if expansions are interactive or automatic
            use_interactive = self.config.get("user_interface", "interactive_mode", True)
            if use_interactive:
                expanded_count, file_decisions, discard_file = self._interactive_expansion(abbreviations, tree, file_path)
            else:
                expanded_count = self._automatic_expansion(abbreviations, tree)
            
            # Handle document discard decision if applicable
            if discard_file:
                discarded_dir = self.config.get("paths", "discarded_directory", 
                                               os.path.join(output_dir, "discarded"))
                os.makedirs(discarded_dir, exist_ok=True)
                dest_path = os.path.join(discarded_dir, os.path.basename(file_path))
                
                keep_decisions = Confirm.ask(
                    "Do you want to include the decisions made so far in the training dataset?", 
                    default=False
                )
                
                if not keep_decisions:
                    # Remove any collected decisions for this file
                    keys_to_remove = []
                    for key, decision in self.user_decisions.items():
                        if decision.get('file_path') == file_path:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.user_decisions[key]
                    
                    console.print(f"[yellow]Removed {len(keys_to_remove)} decisions for this file.[/yellow]")
                else:
                    console.print("[green]Keeping decisions for training dataset.[/green]")
                
                # Copy file to discarded directory
                import shutil
                shutil.copy2(file_path, dest_path)
                console.print(f"[bold]Moved {rel_path} to discarded directory[/bold]")
                return
            
            self.files_processed += 1
            self.abbreviations_expanded += expanded_count
            
            self.logger.info(f"Processed {file_path}, collected {expanded_count} abbreviation expansions")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing {file_path}: {e}")
            self.logger.error(traceback.format_exc())
            console.print(f"[bold red]Error processing {file_path}: {e}[/bold red]")
    
    def _automatic_expansion(self, abbreviations: List[AbbreviationInfo], tree) -> int:
        """
        Automatically expand abbreviations using the highest confidence suggestion.
        
        Args:
            abbreviations: List of abbreviation information objects
            tree: XML tree object
            
        Returns:
            Number of abbreviations expanded
        """
        expanded_count = 0
        
        for abbr in abbreviations:
            # Use normalized_form instead of abbr_text to generate suggestions
            suggestions = self.suggestion_generator.generate_suggestions(
                abbr.normalized_form,
                abbr.context_before,
                abbr.context_after,
                abbr.metadata,
                normalized_abbr=abbr.normalized_form
            )
            
            if not suggestions:
                continue
            
            best_suggestion = suggestions[0]['expansion']
            
            abbr_key = abbr.abbr_text if abbr.abbr_text else abbr.normalized_form
            if not abbr_key:
                abbr_key = f"unknown_{len(self.user_decisions)}"
                
            self.user_decisions[abbr_key] = {
                'expansion': best_suggestion,
                'context_before': abbr.context_before,
                'context_after': abbr.context_after,
                'abbreviation': abbr.abbr_text,
                'source': suggestions[0]['source'],
                'confidence': suggestions[0]['confidence'],
                'file_path': abbr.file_path,
                'metadata': abbr.metadata
            }
            
            expanded_count += 1
        
        return expanded_count
    
    def _clean_display_text(self, text: str) -> str:
        """
        Clean text for display. For abbreviations, do not remove any content
        as we want to show the full word including text after the $ symbol.
        """
        return text
    
    def _interactive_expansion(self, abbreviations: List[AbbreviationInfo], tree, file_path: str) -> tuple:
        """
        Interactively expand abbreviations with user input.
        
        Args:
            abbreviations: List of abbreviation information objects
            tree: XML tree object
            file_path: Path to the file being processed
            
        Returns:
            Tuple containing (Number of abbreviations expanded, file decisions dict, discard flag)
        """
        expanded_count = 0
        file_decisions = {}
        discard_file = False

        for i, abbr in enumerate(abbreviations, 1):
            element_tag = abbr.abbr_element.tag.split('}')[-1] if '}' in abbr.abbr_element.tag else abbr.abbr_element.tag
            # Prefer normalized_form for display if available
            display_text = abbr.normalized_form if abbr.normalized_form else (abbr.abbr_text or "Unknown")
            
            # Clean fused words in the display text
            display_text = self._clean_display_text(display_text)
            
            # If display_text contains multiple words, pick the one with a special marker
            if len(display_text.split()) > 2:
                for word in display_text.split():
                    if '$' in word or any(char in word for char in ['ã', 'ẽ', 'ĩ', 'õ', 'ũ', 'ñ']):
                        display_text = word
                        break
            
            console.print(f"\n[bold]Abbreviation {i}/{len(abbreviations)}:[/bold] [yellow]{display_text}[/yellow] (<{element_tag}> element)")
            
            if abbr.normalized_form:
                normalized_display = abbr.normalized_form
                # Clean fused words in normalized display
                normalized_display = self._clean_display_text(normalized_display)
                
                if len(normalized_display.split()) > 2:
                    for word in normalized_display.split():
                        if '$' in word or any(char in word for char in ['ã', 'ẽ', 'ĩ', 'õ', 'ũ', 'ñ']):
                            normalized_display = word
                            break
                    if len(normalized_display.split()) > 2:
                        normalized_display = "..." + normalized_display.split()[-1]
                
                console.print(f"Normalized for dictionary lookup: [cyan]{normalized_display}[/cyan]")
            
            console.print(f"Location (XPath): [dim]{abbr.xpath}[/dim]")
            
            if abbr.context_before or abbr.context_after:
                context_display = f"[magenta]{abbr.context_before}[/magenta] [bold yellow]{display_text}[/bold yellow] [magenta]{abbr.context_after}[/magenta]"
                console.print(f"Context: {context_display}")
            
            # Generate suggestions using the normalized abbreviation
            suggestions = self.suggestion_generator.generate_suggestions(
                abbr.normalized_form,
                abbr.context_before,
                abbr.context_after,
                abbr.metadata,
                normalized_abbr=abbr.normalized_form
            )
            
            # Remove any single-character suggestions if proper suggestions exist
            proper_suggestions = [s for s in suggestions if len(s['expansion']) > 1]
            if proper_suggestions:
                suggestions = proper_suggestions
            
            # Display the suggestions in a table
            table = Table(title="Expansion Suggestions")
            table.add_column("Option", style="cyan")
            table.add_column("Expansion", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Source", style="blue")
            
            for idx, sugg in enumerate(suggestions, start=1):
                table.add_row(
                    str(idx),
                    sugg['expansion'],
                    f"{sugg['confidence']:.2f}",
                    sugg['source']
                )
            
            # Add options for custom expansion, skipping, and discarding document
            table.add_row("c", "Custom expansion", "-", "-")
            table.add_row("s", "Skip this abbreviation", "-", "-")
            table.add_row("d", "Discard this document", "-", "-")
            
            console.print(table)
            
            try:
                choices = [str(i) for i in range(1, len(suggestions)+1)] + ["c", "s", "d"]
                choice = Prompt.ask("Select an option", choices=choices)
                console.print(f"[bold]You selected:[/bold] {choice}")
            except EOFError:
                console.print("[yellow]Input interrupted. Skipping this abbreviation.[/yellow]")
                continue
            
            # Handle discard document option
            if choice.lower() == "d":
                console.print("[yellow]Discarding this document.[/yellow]")
                discard_file = True
                break
                
            if choice.lower() == "s":
                console.print("Skipping this abbreviation.")
                continue
            
            expansion = None
            source = None
            confidence = None
            
            if choice.lower() == "c":
                try:
                    custom_panel = Panel.fit("Please type your custom expansion and press Enter", title="Custom Expansion", border_style="green")
                    console.print(custom_panel)
                    custom_exp = console.input("Custom Expansion: ").strip()
                    console.print(f"[bold]You entered custom expansion:[/bold] {custom_exp}")
                    if not custom_exp or len(custom_exp) <= 1:
                        console.print("[bold red]Custom expansion must be longer than one character. Skipping this abbreviation.[/bold red]")
                        continue
                    self.suggestion_generator.add_custom_expansion(abbr.abbr_text, custom_exp)
                    self.suggestion_generator.save_user_dictionary()
                    expansion = custom_exp
                    source = "custom"
                    confidence = 1.0
                    console.print(f"[green]Custom expansion '{expansion}' recorded for '{abbr.abbr_text}'.[/green]")
                except EOFError:
                    console.print("[yellow]Input interrupted. Skipping this abbreviation.[/yellow]")
                    continue
            elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                idx = int(choice) - 1
                expansion = suggestions[idx]['expansion']
                source = suggestions[idx]['source']
                confidence = suggestions[idx]['confidence']
            else:
                console.print("[bold red]Invalid choice. Skipping this abbreviation.[/bold red]")
                continue
            
            abbr_key = abbr.abbr_text if abbr.abbr_text else abbr.normalized_form
            if not abbr_key:
                abbr_key = f"unknown_{len(self.user_decisions)}"
            
            decision = {
                'expansion': expansion,
                'context_before': abbr.context_before,
                'context_after': abbr.context_after,
                'abbreviation': abbr.abbr_text,
                'source': source,
                'confidence': confidence,
                'file_path': file_path,
                'metadata': abbr.metadata
            }
            
            self.user_decisions[abbr_key] = decision
            file_decisions[abbr_key] = decision
            
            expanded_count += 1
            console.print(f"[green]Recorded expansion: {expansion}[/green]")
        
        return expanded_count, file_decisions, discard_file
    
    def build_dataset(self):
        """
        Build a dataset from user decisions and/or extracted abbreviations.
        """
        console.print("[bold]Building Dataset[/bold]")
        
        # Let the user choose the dataset format
        format_options = {
            "1": "JSON (Standard format)",
            "2": "JSONL (Line-oriented format for LLM training)",
            "3": "TEI XML (Preserves XML structure with before/after expansion)"
        }
        
        console.print("\n[bold]Dataset Format Options:[/bold]")
        for key, desc in format_options.items():
            console.print(f"  {key}. {desc}")
            
        selected_format = Prompt.ask(
            "\nSelect dataset format", 
            choices=list(format_options.keys()),
            default="1"
        )
        
        # Map selection to actual format string
        format_map = {
            "1": "json",
            "2": "jsonl",
            "3": "tei_xml"
        }
        
        selected_format_type = format_map[selected_format]
        console.print(f"[green]Selected format: {format_options[selected_format]}[/green]")
        
        # Determine whether to use TEI XML processing
        use_tei_xml = selected_format_type == "tei_xml"
        
        if self.user_decisions:
            console.print(f"[green]Using {len(self.user_decisions)} collected abbreviation expansions.[/green]")
            
            if use_tei_xml:
                # For TEI XML format, we need to reprocess the original files to capture XML structure
                input_path = self.config.get("paths", "input_path")
                
                # Extract unique file paths from user decisions
                file_paths = set()
                for _, decision in self.user_decisions.items():
                    if 'file_path' in decision and decision['file_path']:
                        file_paths.add(decision['file_path'])
                
                console.print(f"[green]Found {len(file_paths)} source files for TEI XML processing.[/green]")
                
                if not file_paths:
                    console.print("[yellow]No source files available for TEI XML format. Using standard format instead.[/yellow]")
                    use_tei_xml = False
                
                all_abbreviations = []
                
                if use_tei_xml:
                    from rich.progress import Progress
                    with Progress() as progress:
                        task = progress.add_task("[cyan]Processing source files...", total=len(file_paths))
                        
                        for file_path in file_paths:
                            progress.update(task, description=f"[cyan]Processing {file_path}...[/cyan]")
                            
                            abbreviations, _ = self.tei_processor.parse_document(file_path)
                            
                            # Attach expansion data from user decisions
                            for abbr in abbreviations:
                                abbr_key = abbr.abbr_text if abbr.abbr_text else abbr.normalized_form
                                if abbr_key in self.user_decisions:
                                    abbr.expansion = self.user_decisions[abbr_key]['expansion']
                                else:
                                    # If we can't find the exact match, try a fuzzy match
                                    for key, decision in self.user_decisions.items():
                                        if (decision.get('file_path') == file_path and 
                                            (key in abbr.abbr_text or abbr.abbr_text in key)):
                                            abbr.expansion = decision['expansion']
                                            break
                            
                            # Only keep abbreviations that have expansion data
                            abbreviations = [a for a in abbreviations if hasattr(a, 'expansion')]
                            all_abbreviations.extend(abbreviations)
                            
                            progress.update(task, advance=1)
                    
                    console.print(f"[green]Processed {len(all_abbreviations)} abbreviations with XML structure.[/green]")
                    
                    if all_abbreviations:
                        entries = self.dataset_builder.process_abbreviations_tei(all_abbreviations)
                    else:
                        console.print("[yellow]No abbreviations found with expansion data. Using standard format.[/yellow]")
                        use_tei_xml = False
            
            # Fall back to standard processing if TEI XML processing failed
            if not use_tei_xml:
                entries = []
                for abbr_text, decision in self.user_decisions.items():
                    entry = {
                        'abbreviation': abbr_text,
                        'expansion': decision['expansion'],
                        'context_before': decision.get('context_before', ''),
                        'context_after': decision.get('context_after', ''),
                        'source': {
                            'file': decision.get('file_path', ''),
                            'confidence': decision.get('confidence', 1.0),
                            'source_type': decision.get('source', 'user')
                        }
                    }
                    
                    if 'metadata' in decision and decision['metadata']:
                        entry['metadata'] = decision['metadata']
                        
                    entries.append(entry)
                    
                console.print(f"[green]Created {len(entries)} entries from user decisions.[/green]")
        else:
            input_path = self.config.get("paths", "input_path")
            output_path = self.config.get("paths", "output_path")
            
            xml_files = []
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".xml"):
                        xml_files.append(os.path.join(root, file))
            
            if not xml_files:
                console.print("[yellow]No XML files found in the input directory.[/yellow]")
                return
            
            console.print(f"[bold]Found {len(xml_files)} XML files to process.[/bold]")
            
            all_abbreviations = []
            
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("[cyan]Extracting abbreviations...", total=len(xml_files))
                
                for file_path in xml_files:
                    rel_path = os.path.relpath(file_path, input_path)
                    progress.update(task, description=f"[cyan]Processing {rel_path}...[/cyan]")
                    
                    abbreviations, _ = self.tei_processor.parse_document(file_path)
                    all_abbreviations.extend(abbreviations)
                    
                    progress.update(task, advance=1)
            
            console.print(f"[green]Extracted {len(all_abbreviations)} abbreviations.[/green]")
            
            if use_tei_xml:
                entries = self.dataset_builder.process_abbreviations_tei(all_abbreviations)
            else:
                entries = self.dataset_builder.process_abbreviations(all_abbreviations)
        
        if not entries:
            console.print("[yellow]No entries to include in dataset.[/yellow]")
            return
            
        train_set, val_set, test_set = self.dataset_builder.split_dataset(entries)
        
        output_path = self.config.get("paths", "output_path")
        dataset_dir = os.path.join(output_path, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"expansion_dataset_{timestamp}"
        
        # Save datasets with the selected format
        self.dataset_builder.save_dataset(
            train_set, 
            os.path.join(dataset_dir, f"{base_filename}_train.json"),
            format=selected_format_type
        )
        
        self.dataset_builder.save_dataset(
            val_set, 
            os.path.join(dataset_dir, f"{base_filename}_validation.json"),
            format=selected_format_type
        )
        
        self.dataset_builder.save_dataset(
            test_set, 
            os.path.join(dataset_dir, f"{base_filename}_test.json"),
            format=selected_format_type
        )
        
        # Only create formatted training file for non-TEI XML formats
        if not use_tei_xml:
            system_message = self.config.get("language_model_integration", "openai", {}).get(
                "system_message", 
                "You are a linguist specializing in early modern texts. Your task is to expand abbreviated words."
            )
            
            formatted_train = self.dataset_builder.format_for_llm_training(train_set, system_message)
            self.dataset_builder.save_dataset(
                formatted_train, 
                os.path.join(dataset_dir, f"{base_filename}_train_formatted.jsonl"), 
                format="jsonl"
            )
        
        console.print(f"[bold green]Dataset creation complete![/bold green]")
        console.print(f"Train set: {len(train_set)} entries")
        console.print(f"Validation set: {len(val_set)} entries")
        console.print(f"Test set: {len(test_set)} entries")
    
    def interactive_expansion(self):
        """
        Interactive abbreviation expansion without document processing.
        """
        console.print("[bold]Interactive Abbreviation Expansion[/bold]")
        console.print("Enter abbreviations to expand them. Type 'exit' to return to the main menu.")
        
        while True:
            abbr_text = Prompt.ask("\nEnter abbreviation", default="exit")
            
            if abbr_text.lower() == 'exit':
                break
            
            context = Prompt.ask("Enter context (optional)")
            
            normalized_abbr = None
            try:
                from modules.unicode_replacement import UnicodeReplacement
                normalized_abbr = UnicodeReplacement.normalize_abbreviation(abbr_text)
                if normalized_abbr != abbr_text:
                    console.print(f"Normalized for dictionary lookup: [cyan]{normalized_abbr}[/cyan]")
            except (ImportError, AttributeError):
                pass
            
            suggestions = self.suggestion_generator.generate_suggestions(
                normalized_abbr if normalized_abbr else abbr_text,
                context_before=context,
                context_after="",
                normalized_abbr=normalized_abbr
            )
            
            suggestions = [s for s in suggestions if len(s['expansion']) > 1]
            
            if not suggestions:
                console.print("[yellow]No suggestions available for this abbreviation.[/yellow]")
                continue
            
            table = Table(title=f"Suggestions for '{abbr_text}'")
            table.add_column("Expansion", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Source", style="blue")
            
            for sugg in suggestions:
                expansion_text = sugg['expansion']
                if len(expansion_text) > 50:
                    expansion_text = expansion_text[:47] + "..."
                table.add_row(
                    expansion_text,
                    f"{sugg['confidence']:.2f}",
                    sugg['source']
                )
            
            console.print(table)
            
            choice = Prompt.ask("Enter the expansion, or type 'c' for custom, 's' to skip")
            if choice.lower() == "s":
                continue
            elif choice.lower() == "c":
                custom_panel = Panel.fit("Please type your custom expansion and press Enter", title="Custom Expansion", border_style="green")
                console.print(custom_panel)
                custom_exp = console.input("Custom Expansion: ").strip()
                console.print(f"[bold]You entered custom expansion:[/bold] {custom_exp}")
                if not custom_exp or len(custom_exp) <= 1:
                    console.print("[bold red]Custom expansion must be longer than one character. Skipping this abbreviation.[/bold red]")
                    continue
                self.suggestion_generator.add_custom_expansion(abbr_text, custom_exp)
                self.suggestion_generator.save_user_dictionary()
                expansion = custom_exp
                source = "custom"
                confidence = 1.0
                console.print(f"[green]Custom expansion '{expansion}' recorded for '{abbr_text}'.[/green]")
            else:
                expansion = choice
                source = "manual"
                confidence = 1.0
            
            self.user_decisions[abbr_text] = {
                'expansion': expansion,
                'context_before': context,
                'context_after': "",
                'abbreviation': abbr_text,
                'source': source,
                'confidence': confidence,
                'file_path': "",
                'metadata': {}
            }
            
            console.print(f"[green]Recorded expansion: {expansion}[/green]")
    
    def show_statistics(self):
        """Display statistics about processed documents and abbreviations."""
        console.print("[bold]Statistics[/bold]")
        
        # Create default statistics if TEIProcessor instance is not available
        try:
            from tei.processor import TEIProcessor
            tei_processor = TEIProcessor(self.config)
            tei_stats = tei_processor.stats
        except (ImportError, AttributeError) as e:
            console.print(f"Error: {e}")
            tei_stats = {
                'documents_processed': 0,
                'abbreviations_found': 0,
                'already_expanded': 0,
                'malformed_abbr': 0
            }
        
        table = Table(title="Processing Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents Processed", str(tei_stats.get('documents_processed', 0)))
        table.add_row("Abbreviations Found", str(tei_stats.get('abbreviations_found', 0)))
        table.add_row("Already Expanded", str(tei_stats.get('already_expanded', 0)))
        table.add_row("Malformed Abbreviations", str(tei_stats.get('malformed_abbr', 0)))
        table.add_row("Current Session Expansions", str(getattr(self, 'abbreviations_expanded', 0)))
        
        console.print(table)
        
        # Get suggestion statistics if available
        try:
            sugg_stats = self.suggestion_generator.get_statistics()
        except AttributeError:
            sugg_stats = {
                'total_suggestions': 0,
                'dictionary_matches': 0,
                'pattern_matches': 0,
                'wordnet_suggestions': 0,
                'lm_suggestions': 0,
                'failed_abbreviations': 0
            }
        
        table = Table(title="Suggestion Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Suggestions Generated", str(sugg_stats.get('total_suggestions', 0)))
        table.add_row("Dictionary Matches", str(sugg_stats.get('dictionary_matches', 0)))
        table.add_row("Pattern Matches", str(sugg_stats.get('pattern_matches', 0)))
        table.add_row("WordNet Suggestions", str(sugg_stats.get('wordnet_suggestions', 0)))
        table.add_row("Language Model Suggestions", str(sugg_stats.get('lm_suggestions', 0)))
        table.add_row("Failed Abbreviations", str(sugg_stats.get('failed_abbreviations', 0)))
        
        console.print(table)
        
        # Get dataset statistics if available
        try:
            dataset_stats = self.dataset_builder.get_statistics()
        except AttributeError:
            dataset_stats = {
                'total_entries': 0,
                'train_entries': 0,
                'validation_entries': 0,
                'test_entries': 0
            }
        
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Entries", str(dataset_stats.get('total_entries', 0)))
        table.add_row("Training Entries", str(dataset_stats.get('train_entries', 0)))
        table.add_row("Validation Entries", str(dataset_stats.get('validation_entries', 0)))
        table.add_row("Test Entries", str(dataset_stats.get('test_entries', 0)))
        table.add_row("Skipped Entries", str(dataset_stats.get('skipped_entries', 0)))
        table.add_row("Duplicate Entries", str(dataset_stats.get('duplicate_entries', 0)))
        
        console.print(table)
    
    def show_settings(self):
        """Display and modify settings."""
        console.print("[bold]Settings[/bold]")
        
        settings = {
            "User Interface": {
                "interactive_mode": self.config.get("user_interface", "interactive_mode", True),
                "show_confidence_scores": self.config.get("user_interface", "show_confidence_scores", True)
            },
            "XML Processing": {
                "use_choice_tags": self.config.get("xml_processing", "use_choice_tags", False),
                "add_xml_ids": self.config.get("xml_processing", "add_xml_ids", True),
                "skip_expanded": self.config.get("settings", "skip_expanded", False)
            },
            "Language Model": {
                "enabled": self.config.get("language_model_integration", "enabled", True),
                "provider": self.config.get("language_model_integration", "provider", "openai"),
                "model_name": self.config.get("language_model_integration", "model_name", "gpt-4")
            },
            "Dataset": {
                "format": self.config.get("dataset", "format", "json"),
                "include_metadata": self.config.get("dataset", "include_metadata", True),
                "train_ratio": self.config.get("dataset", "train_ratio", 0.8)
            }
        }
        
        table = Table(title="Current Settings")
        table.add_column("Category", style="cyan")
        table.add_column("Setting", style="blue")
        table.add_column("Value", style="green")
        
        for category, category_settings in settings.items():
            for setting, value in category_settings.items():
                table.add_row(category, setting, str(value))
        
        console.print(table)
        
        # Offer to change dataset format
        change_format = Confirm.ask("\nDo you want to change the dataset format?", default=False)
        
        if change_format:
            format_options = {
                "1": "JSON (Standard format)",
                "2": "JSONL (Line-oriented format for LLM training)",
                "3": "TEI XML (Preserves XML structure with before/after expansion)"
            }
            
            console.print("\n[bold]Dataset Format Options:[/bold]")
            for key, desc in format_options.items():
                console.print(f"  {key}. {desc}")
                
            selected_format = Prompt.ask(
                "\nSelect dataset format", 
                choices=list(format_options.keys()),
                default="1"
            )
            
            # Map selection to actual format string
            format_map = {
                "1": "json",
                "2": "jsonl",
                "3": "tei_xml"
            }
            
            selected_format_type = format_map[selected_format]
            
            # Update the config
            self.config.set("dataset", "format", selected_format_type)
            console.print(f"[green]Dataset format updated to: {format_options[selected_format]}[/green]")
            
            # Show updated settings table
            settings["Dataset"]["format"] = selected_format_type
            
            updated_table = Table(title="Updated Settings")
            updated_table.add_column("Category", style="cyan")
            updated_table.add_column("Setting", style="blue")
            updated_table.add_column("Value", style="green")
            
            for category, category_settings in settings.items():
                for setting, value in category_settings.items():
                    updated_table.add_row(category, setting, str(value))
            
            console.print(updated_table)
        else:
            console.print("\n[yellow]No changes made to settings.[/yellow]")
    
    def run(self):
        """Run the main application loop."""
        self.show_welcome()
        
        while True:
            choice = self.show_main_menu()
            
            try:
                if choice == "1":
                    self.process_tei_documents()
                elif choice == "2":
                    self.interactive_expansion()
                elif choice == "3":
                    self.build_dataset()
                elif choice == "4":
                    self.show_statistics()
                elif choice == "5":
                    self.show_settings()
                elif choice == "6":
                    if self.user_decisions:
                        try:
                            if Confirm.ask("Save your work before exiting?"):
                                self._save_user_decisions()
                        except EOFError:
                            console.print("[yellow]Input interrupted. Exiting without saving.[/yellow]")
                    console.print("[bold green]Thank you for using Amanuensis 2.0![/bold green]")
                    break
            except EOFError:
                console.print("[yellow]Input interrupted. Returning to main menu.[/yellow]")
                continue


if __name__ == '__main__':
    ui = UserInterface()
    ui.run()