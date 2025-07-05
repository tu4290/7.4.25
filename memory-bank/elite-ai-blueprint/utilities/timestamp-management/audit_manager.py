#!/usr/bin/env python3
"""
AI Timestamp Audit Manager

Manages timestamp auditing for AI cognitive systems, providing detailed
tracking of AI session activities, memory bank updates, and cognitive
process timing. This is completely separate from trading system timing.

Author: Elite AI Blueprint
Version: 1.0.0
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


class AITimestampAuditManager:
    """Manages timestamp auditing for AI cognitive systems."""
    
    def __init__(self, audit_file_path: Optional[str] = None):
        """Initialize the AI Timestamp Audit Manager.
        
        Args:
            audit_file_path: Custom path for audit file. If None, uses default.
        """
        self.audit_file = audit_file_path or self._get_default_audit_path()
        self.session_id = self._generate_session_id()
        
    def _get_default_audit_path(self) -> str:
        """Get default audit file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "audit.json")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID for AI activities."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"ai_session_{timestamp}"
    
    def current_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def log_ai_activity(self, activity_type: str, details: Dict[str, Any]) -> None:
        """Log AI cognitive activity to audit trail.
        
        Args:
            activity_type: Type of AI activity (e.g., 'memory_update', 'cognitive_process')
            details: Activity details and metadata
        """
        audit_entry = {
            "timestamp": self.current_timestamp(),
            "session_id": self.session_id,
            "activity_type": activity_type,
            "details": details,
            "source": "ai_cognitive_system"
        }
        
        self._append_to_audit_log(audit_entry)
    
    def log_memory_bank_update(self, file_path: str, update_type: str, 
                              metadata: Optional[Dict] = None) -> None:
        """Log memory bank file updates.
        
        Args:
            file_path: Path to the updated file
            update_type: Type of update (e.g., 'created', 'modified', 'accessed')
            metadata: Additional metadata about the update
        """
        details = {
            "file_path": file_path,
            "update_type": update_type,
            "metadata": metadata or {}
        }
        
        self.log_ai_activity("memory_bank_update", details)
    
    def log_cognitive_process(self, process_name: str, duration_ms: float,
                            success: bool, details: Optional[Dict] = None) -> None:
        """Log cognitive process execution.
        
        Args:
            process_name: Name of the cognitive process
            duration_ms: Process duration in milliseconds
            success: Whether the process completed successfully
            details: Additional process details
        """
        process_details = {
            "process_name": process_name,
            "duration_ms": duration_ms,
            "success": success,
            "details": details or {}
        }
        
        self.log_ai_activity("cognitive_process", process_details)
    
    def _append_to_audit_log(self, entry: Dict) -> None:
        """Append entry to audit log file.
        
        Args:
            entry: Audit entry to append
        """
        # Load existing audit log
        audit_log = self._load_audit_log()
        
        # Append new entry
        audit_log.append(entry)
        
        # Keep only last 5000 entries for AI systems
        if len(audit_log) > 5000:
            audit_log = audit_log[-5000:]
        
        # Save audit log
        self._save_audit_log(audit_log)
    
    def _load_audit_log(self) -> List[Dict]:
        """Load existing audit log.
        
        Returns:
            List of audit entries
        """
        if not os.path.exists(self.audit_file):
            return []
        
        try:
            with open(self.audit_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_audit_log(self, audit_log: List[Dict]) -> None:
        """Save audit log to file.
        
        Args:
            audit_log: List of audit entries to save
        """
        os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
        with open(self.audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_log, f, indent=2, ensure_ascii=False)
    
    def get_recent_activity(self, limit: int = 50, 
                          activity_type: Optional[str] = None) -> List[Dict]:
        """Get recent AI activity from audit log.
        
        Args:
            limit: Maximum number of entries to return
            activity_type: Filter by specific activity type
            
        Returns:
            List of recent audit entries
        """
        audit_log = self._load_audit_log()
        
        if activity_type:
            audit_log = [entry for entry in audit_log 
                        if entry.get('activity_type') == activity_type]
        
        return audit_log[-limit:] if audit_log else []
    
    def get_session_activity(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get all activity for a specific AI session.
        
        Args:
            session_id: Session ID to filter by. If None, uses current session.
            
        Returns:
            List of session audit entries
        """
        target_session = session_id or self.session_id
        audit_log = self._load_audit_log()
        
        return [entry for entry in audit_log 
                if entry.get('session_id') == target_session]
    
    def generate_activity_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate activity report for the last N hours.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Activity report with statistics
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        audit_log = self._load_audit_log()
        
        # Filter entries within time range
        recent_entries = [
            entry for entry in audit_log
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) >= cutoff_time
        ]
        
        # Generate statistics
        activity_counts = {}
        session_counts = {}
        
        for entry in recent_entries:
            activity_type = entry.get('activity_type', 'unknown')
            session_id = entry.get('session_id', 'unknown')
            
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
        
        return {
            "time_range_hours": hours,
            "total_activities": len(recent_entries),
            "activity_breakdown": activity_counts,
            "session_breakdown": session_counts,
            "unique_sessions": len(session_counts),
            "report_generated": self.current_timestamp()
        }
    
    def clear_old_entries(self, days: int = 30) -> int:
        """Clear audit entries older than specified days.
        
        Args:
            days: Number of days to retain
            
        Returns:
            Number of entries removed
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        audit_log = self._load_audit_log()
        
        original_count = len(audit_log)
        
        # Keep only recent entries
        filtered_log = [
            entry for entry in audit_log
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) >= cutoff_time
        ]
        
        self._save_audit_log(filtered_log)
        
        return original_count - len(filtered_log)


if __name__ == "__main__":
    # Example usage
    audit_manager = AITimestampAuditManager()
    
    # Log some example activities
    audit_manager.log_memory_bank_update(
        "/path/to/memory/file.md", 
        "modified", 
        {"size_bytes": 1024, "sections_updated": ["progress", "context"]}
    )
    
    audit_manager.log_cognitive_process(
        "pattern_recognition", 
        150.5, 
        True, 
        {"patterns_found": 3, "confidence": 0.85}
    )
    
    # Generate activity report
    report = audit_manager.generate_activity_report(24)
    print(json.dumps(report, indent=2))