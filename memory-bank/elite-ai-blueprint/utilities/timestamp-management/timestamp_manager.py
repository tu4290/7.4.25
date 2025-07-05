#!/usr/bin/env python3
"""
Timestamp Manager v2.5
Automated timestamp management for Memory Bank system

This module provides automated, accurate timestamp generation and validation
for the Elite Options System Memory Bank, ensuring consistent and reliable
timestamp tracking across all memory bank operations.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import hashlib

class MemoryBankTimestamp:
    """Automated timestamp management for Memory Bank system."""
    
    def __init__(self, memory_bank_path: str = None):
        """Initialize timestamp manager.
        
        Args:
            memory_bank_path: Path to memory bank directory
        """
        self.memory_bank_path = memory_bank_path or self._get_default_memory_bank_path()
        self.session_id = self._generate_session_id()
        
    def _get_default_memory_bank_path(self) -> str:
        """Get default memory bank path."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "memory-bank" / "elite-options-system")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now(timezone.utc)
        session_hash = hashlib.md5(f"{timestamp.isoformat()}".encode()).hexdigest()[:8]
        return f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_hash}"
    
    @staticmethod
    def current_timestamp() -> str:
        """Get current ISO 8601 timestamp with timezone."""
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def current_timestamp_local() -> str:
        """Get current local timestamp."""
        return datetime.now().isoformat()
    
    @staticmethod
    def format_milestone_date() -> str:
        """Format current date for milestone entries."""
        return datetime.now().strftime("%B %d, %Y")
    
    @staticmethod
    def format_display_timestamp() -> str:
        """Format timestamp for display in memory bank."""
        return datetime.now().strftime("%B %d, %Y %H:%M:%S")
    
    def validate_timestamp(self, timestamp_str: str) -> Tuple[bool, str]:
        """Validate timestamp against current system time.
        
        Args:
            timestamp_str: Timestamp string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse the timestamp
            if 'T' in timestamp_str:
                parsed_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                parsed_time = datetime.strptime(timestamp_str, "%B %d, %Y")
            
            current_time = datetime.now(timezone.utc)
            
            # Check if timestamp is in the future (with 1 hour tolerance)
            current_naive = current_time.replace(tzinfo=None)
            if parsed_time > current_naive + timedelta(hours=1):
                return False, f"Timestamp {timestamp_str} is in the future"
            
            # Check if timestamp is too old (more than 5 years)
            five_years_ago = current_time - timedelta(days=5*365)
            if parsed_time < five_years_ago.replace(tzinfo=None):
                return False, f"Timestamp {timestamp_str} is more than 5 years old"
            
            return True, "Valid timestamp"
            
        except Exception as e:
            return False, f"Invalid timestamp format: {str(e)}"
    
    def create_milestone_entry(self, milestone_name: str, details: List[str]) -> Dict:
        """Create a properly formatted milestone entry.
        
        Args:
            milestone_name: Name of the milestone
            details: List of detail strings
            
        Returns:
            Dictionary containing formatted milestone entry
        """
        timestamp = self.current_timestamp()
        display_date = self.format_milestone_date()
        
        entry = {
            "milestone_name": milestone_name,
            "display_date": display_date,
            "system_timestamp": timestamp,
            "session_id": self.session_id,
            "details": details,
            "validation_status": "✅ System Generated"
        }
        
        # Log to audit trail
        self._log_to_audit("milestone_created", entry)
        
        return entry
    
    def format_milestone_markdown(self, milestone_name: str, details: List[str]) -> str:
        """Format milestone entry for markdown insertion.
        
        Args:
            milestone_name: Name of the milestone
            details: List of detail strings
            
        Returns:
            Formatted markdown string
        """
        entry = self.create_milestone_entry(milestone_name, details)
        
        markdown = f"- **✅ {milestone_name} ({entry['display_date']})**\n"
        markdown += f"  - **System Timestamp**: {entry['system_timestamp']}\n"
        markdown += f"  - **Session ID**: {entry['session_id']}\n"
        
        for detail in details:
            markdown += f"  - **{detail.split(':')[0]}**: {':'.join(detail.split(':')[1:]).strip()}\n"
        
        return markdown
    
    def update_file_timestamp_metadata(self, file_path: str) -> Dict:
        """Update timestamp metadata for a memory bank file.
        
        Args:
            file_path: Path to the memory bank file
            
        Returns:
            Dictionary containing timestamp metadata
        """
        timestamp = self.current_timestamp()
        
        metadata = {
            "last_updated": timestamp,
            "session_id": self.session_id,
            "validation_status": "✅ Verified",
            "source": "System Generated",
            "file_path": file_path
        }
        
        return metadata
    

    

    

    
    def validate_memory_bank_timestamps(self) -> Dict[str, List[str]]:
        """Validate all timestamps in memory bank files.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid_files": [],
            "invalid_timestamps": [],
            "missing_timestamps": [],
            "errors": []
        }
        
        memory_bank_files = [
            "activeContext.md",
            "progress.md",
            "productContext.md",
            "systemPatterns.md",
            "techContext.md",
            "projectbrief.md"
        ]
        
        for file_name in memory_bank_files:
            file_path = os.path.join(self.memory_bank_path, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for timestamp patterns
                    import re
                    timestamp_patterns = [
                        r'\(([A-Za-z]+ \d{1,2}, \d{4})\)',  # (Month DD, YYYY)
                        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
                        r'\*\*Last Updated\*\*: ([^\n]+)'  # Metadata format
                    ]
                    
                    found_timestamps = []
                    for pattern in timestamp_patterns:
                        matches = re.findall(pattern, content)
                        found_timestamps.extend(matches)
                    
                    if found_timestamps:
                        # Validate each timestamp
                        all_valid = True
                        for ts in found_timestamps:
                            is_valid, error = self.validate_timestamp(ts)
                            if not is_valid:
                                results["invalid_timestamps"].append(f"{file_name}: {error}")
                                all_valid = False
                        
                        if all_valid:
                            results["valid_files"].append(file_name)
                    else:
                        results["missing_timestamps"].append(file_name)
                        
                except Exception as e:
                    results["errors"].append(f"{file_name}: {str(e)}")
            else:
                results["errors"].append(f"{file_name}: File not found")
        
        return results


# Convenience functions for easy import
def get_current_timestamp() -> str:
    """Get current timestamp - convenience function."""
    return MemoryBankTimestamp.current_timestamp()

def get_milestone_date() -> str:
    """Get formatted milestone date - convenience function."""
    return MemoryBankTimestamp.format_milestone_date()

def create_milestone(name: str, details: List[str]) -> str:
    """Create formatted milestone entry - convenience function."""
    manager = MemoryBankTimestamp()
    return manager.format_milestone_markdown(name, details)

def validate_timestamp(timestamp_str: str) -> Tuple[bool, str]:
    """Validate timestamp - convenience function."""
    manager = MemoryBankTimestamp()
    return manager.validate_timestamp(timestamp_str)


if __name__ == "__main__":
    # Test the timestamp manager
    manager = MemoryBankTimestamp()
    
    print("=== Timestamp Manager Test ===")
    print(f"Current Timestamp: {manager.current_timestamp()}")
    print(f"Milestone Date: {manager.format_milestone_date()}")
    print(f"Session ID: {manager.session_id}")
    
    # Test milestone creation
    test_milestone = manager.format_milestone_markdown(
        "Test Integration",
        [
            "Feature: Timestamp management system",
            "Status: Successfully implemented",
            "Impact: Automated timestamp accuracy"
        ]
    )
    print(f"\nTest Milestone:\n{test_milestone}")
    
    # Test validation
    validation_results = manager.validate_memory_bank_timestamps()
    print(f"\nValidation Results: {validation_results}")