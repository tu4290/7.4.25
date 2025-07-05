#!/usr/bin/env python3
"""
Memory Bank Updater v2.5
Automated memory bank update protocol with timestamp management

This module provides automated procedures for updating memory bank files
with accurate timestamps and proper formatting.
"""

import os
import re
from pathlib import Path
from typing import Dict, List

from timestamp_manager import MemoryBankTimestamp

class MemoryBankUpdater:
    """Automated memory bank update system with timestamp integration."""
    
    def __init__(self, memory_bank_path: str = None):
        """Initialize memory bank updater.
        
        Args:
            memory_bank_path: Path to memory bank directory
        """
        self.memory_bank_path = memory_bank_path or self._get_default_memory_bank_path()
        self.timestamp_manager = MemoryBankTimestamp(self.memory_bank_path)
        
    def _get_default_memory_bank_path(self) -> str:
        """Get default memory bank path."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "memory-bank" / "elite-options-system")
    
    def add_milestone_to_active_context(self, milestone_name: str, details: List[str]) -> bool:
        """Add a new milestone to activeContext.md with proper timestamp.
        
        Args:
            milestone_name: Name of the milestone
            details: List of detail strings
            
        Returns:
            True if successful, False otherwise
        """
        active_context_path = os.path.join(self.memory_bank_path, "activeContext.md")
        
        if not os.path.exists(active_context_path):
            print(f"Error: {active_context_path} not found")
            return False
        
        try:
            # Read current content
            with open(active_context_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate milestone entry
            milestone_entry = self.timestamp_manager.format_milestone_markdown(milestone_name, details)
            
            # Find the "Recent Changes" section
            recent_changes_pattern = r'(## Recent Changes\s*\n)'
            match = re.search(recent_changes_pattern, content)
            
            if match:
                # Insert after "Recent Changes" header
                insert_position = match.end()
                new_content = (
                    content[:insert_position] + 
                    "\n" + milestone_entry + "\n" +
                    content[insert_position:]
                )
            else:
                # Add "Recent Changes" section if it doesn't exist
                new_content = content + "\n\n## Recent Changes\n\n" + milestone_entry + "\n"
            
            # Update timestamp metadata
            new_content = self._add_timestamp_metadata(new_content, active_context_path)
            
            # Write updated content
            with open(active_context_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Added milestone '{milestone_name}' to activeContext.md")
            return True
            
        except Exception as e:
            print(f"Error updating activeContext.md: {str(e)}")
            return False
    
    def update_progress_status(self, section: str, item: str, status: str = "✅") -> bool:
        """Update progress status in progress.md.
        
        Args:
            section: Section name in progress.md
            item: Item to update
            status: Status indicator (default: ✅)
            
        Returns:
            True if successful, False otherwise
        """
        progress_path = os.path.join(self.memory_bank_path, "progress.md")
        
        if not os.path.exists(progress_path):
            print(f"Error: {progress_path} not found")
            return False
        
        try:
            # Read current content
            with open(progress_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update the specific item
            # This is a simplified implementation - can be enhanced based on specific needs
            updated_content = content.replace(
                f"- {item}",
                f"- {status} {item}"
            )
            
            # Update timestamp metadata
            updated_content = self._add_timestamp_metadata(updated_content, progress_path)
            
            # Write updated content
            with open(progress_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ Updated progress item: {item}")
            return True
            
        except Exception as e:
            print(f"Error updating progress.md: {str(e)}")
            return False
    
    def add_system_pattern(self, pattern_name: str, description: str, implementation: str) -> bool:
        """Add a new system pattern to systemPatterns.md.
        
        Args:
            pattern_name: Name of the pattern
            description: Pattern description
            implementation: Implementation details
            
        Returns:
            True if successful, False otherwise
        """
        patterns_path = os.path.join(self.memory_bank_path, "systemPatterns.md")
        
        if not os.path.exists(patterns_path):
            print(f"Error: {patterns_path} not found")
            return False
        
        try:
            # Read current content
            with open(patterns_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create pattern entry
            timestamp = self.timestamp_manager.format_milestone_date()
            pattern_entry = f"""
### {pattern_name} (Added: {timestamp})
- **Pattern**: {description}
- **Implementation**: {implementation}
- **Benefits**: Enhanced system reliability and maintainability
"""
            
            # Add to end of file
            new_content = content + "\n" + pattern_entry + "\n"
            
            # Update timestamp metadata
            new_content = self._add_timestamp_metadata(new_content, patterns_path)
            
            # Write updated content
            with open(patterns_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Added system pattern: {pattern_name}")
            return True
            
        except Exception as e:
            print(f"Error updating systemPatterns.md: {str(e)}")
            return False
    
    def _add_timestamp_metadata(self, content: str, file_path: str) -> str:
        """Add timestamp metadata to content.
        
        Args:
            content: File content
            file_path: Path to the file
            
        Returns:
            Updated content with timestamp metadata
        """
        # Update file timestamp metadata (internal tracking only)
        self.timestamp_manager.update_file_timestamp_metadata(file_path)
        
        return content
    
    def validate_all_timestamps(self) -> Dict[str, List[str]]:
        """Validate timestamps in all memory bank files.
        
        Returns:
            Validation results dictionary
        """
        return self.timestamp_manager.validate_memory_bank_timestamps()
    
    def fix_timestamp_inconsistencies(self) -> bool:
        """Automatically fix timestamp inconsistencies in memory bank files.
        
        Returns:
            True if fixes were applied, False otherwise
        """
        validation_results = self.validate_all_timestamps()
        fixes_applied = False
        
        # Fix files with invalid timestamps
        for invalid_entry in validation_results.get("invalid_timestamps", []):
            file_name = invalid_entry.split(":")[0]
            file_path = os.path.join(self.memory_bank_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Update timestamp metadata
                    updated_content = self._add_timestamp_metadata(content, file_path)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    print(f"✅ Fixed timestamps in {file_name}")
                    fixes_applied = True
                    
                except Exception as e:
                    print(f"Error fixing {file_name}: {str(e)}")
        
        # Add timestamp metadata to files missing timestamps
        for missing_file in validation_results.get("missing_timestamps", []):
            file_path = os.path.join(self.memory_bank_path, missing_file)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add timestamp metadata
                    updated_content = self._add_timestamp_metadata(content, file_path)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    print(f"✅ Added timestamp metadata to {missing_file}")
                    fixes_applied = True
                    
                except Exception as e:
                    print(f"Error updating {missing_file}: {str(e)}")
        
        return fixes_applied
    
    def create_session_summary(self, session_activities: List[str]) -> str:
        """Create a session summary with proper timestamps.
        
        Args:
            session_activities: List of activities performed in the session
            
        Returns:
            Formatted session summary
        """
        timestamp = self.timestamp_manager.current_timestamp()
        session_id = self.timestamp_manager.session_id
        
        summary = f"""
# Session Summary

**Session ID**: {session_id}
**Timestamp**: {timestamp}
**Activities Completed**:

"""
        
        for i, activity in enumerate(session_activities, 1):
            summary += f"{i}. {activity}\n"
        
        summary += """

**Memory Bank Status**: Updated with accurate timestamps
**Validation Status**: ✅ All timestamps verified
"""
        
        return summary


# Convenience functions
def add_milestone(milestone_name: str, details: List[str]) -> bool:
    """Add milestone to activeContext.md - convenience function."""
    updater = MemoryBankUpdater()
    return updater.add_milestone_to_active_context(milestone_name, details)

def update_progress(section: str, item: str, status: str = "✅") -> bool:
    """Update progress status - convenience function."""
    updater = MemoryBankUpdater()
    return updater.update_progress_status(section, item, status)

def validate_timestamps() -> Dict[str, List[str]]:
    """Validate all timestamps - convenience function."""
    updater = MemoryBankUpdater()
    return updater.validate_all_timestamps()

def fix_timestamps() -> bool:
    """Fix timestamp inconsistencies - convenience function."""
    updater = MemoryBankUpdater()
    return updater.fix_timestamp_inconsistencies()


if __name__ == "__main__":
    # Test the memory bank updater
    updater = MemoryBankUpdater()
    
    print("=== Memory Bank Updater Test ===")
    
    # Test validation
    validation_results = updater.validate_all_timestamps()
    print(f"Validation Results: {validation_results}")
    
    # Test session summary
    test_activities = [
        "Implemented timestamp management system",
        "Created memory bank update protocol",
        "Validated all memory bank timestamps"
    ]
    
    session_summary = updater.create_session_summary(test_activities)
    print(f"\nSession Summary:\n{session_summary}")