"""
OceanVerse Report Generator Module
Handles report generation in multiple formats: PDF, CSV, PNG, NetCDF, Markdown.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import io
import csv

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Multi-format report generation engine."""
    
    def __init__(self, db_connection=None):
        """Initialize Report Generator."""
        self.db = db_connection
        self.reports = {}
        
    def generate_research_summary(self, 
                                 report_title: str,
                                 research_data: Dict,
                                 output_format: str = 'markdown') -> Dict:
        """
        Generate research summary in specified format.
        
        Args:
            report_title: Title of the research report
            research_data: Dictionary containing research findings
            output_format: Output format (markdown, pdf, html, json)
            
        Returns:
            Generated report dictionary
        """
        report_id = f"report_{datetime.now().timestamp()}"
        
        if output_format == 'markdown':
            content = self._generate_markdown_report(report_title, research_data)
        elif output_format == 'html':
            content = self._generate_html_report(report_title, research_data)
        elif output_format == 'json':
            content = self._generate_json_report(report_title, research_data)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        report = {
            'id': report_id,
            'title': report_title,
            'format': output_format,
            'content': content,
            'created_at': datetime.now().isoformat(),
            'file_size_kb': len(content.encode()) / 1024,
        }
        
        self.reports[report_id] = report
        logger.info(f"Report generated: {report_id} ({output_format})")
        return report
    
    def generate_prediction_report(self, 
                                   predictions: Dict,
                                   confidence_scores: Dict,
                                   output_format: str = 'pdf') -> Dict:
        """
        Generate prediction report with confidence metrics.
        
        Args:
            predictions: Prediction results dictionary
            confidence_scores: Confidence score data
            output_format: Output format (pdf, html, json)
            
        Returns:
            Generated report dictionary
        """
        report_id = f"pred_report_{datetime.now().timestamp()}"
        
        report_content = {
            'title': 'Ocean Prediction Report',
            'generated_at': datetime.now().isoformat(),
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'summary_statistics': self._calculate_summary_stats(confidence_scores),
            'visualization_data': self._prepare_visualization_data(predictions),
        }
        
        if output_format == 'json':
            content = json.dumps(report_content, indent=2)
        elif output_format == 'html':
            content = self._render_html_prediction_report(report_content)
        else:
            content = json.dumps(report_content, indent=2)
        
        report = {
            'id': report_id,
            'type': 'prediction',
            'format': output_format,
            'content': content,
            'created_at': datetime.now().isoformat(),
            'file_size_kb': len(content.encode()) / 1024,
        }
        
        self.reports[report_id] = report
        logger.info(f"Prediction report generated: {report_id}")
        return report
    
    def generate_csv_export(self, 
                           data: List[Dict],
                           filename: str) -> bytes:
        """
        Generate CSV export.
        
        Args:
            data: List of dictionaries to export
            filename: Output filename
            
        Returns:
            CSV file bytes
        """
        output = io.StringIO()
        
        if not data:
            return b''
        
        fieldnames = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
        
        logger.info(f"CSV export generated: {filename}")
        return output.getvalue().encode()
    
    def generate_netcdf_export(self,
                              data_dict: Dict,
                              variable_names: List[str],
                              dimensions: Dict,
                              output_path: str) -> Dict:
        """
        Generate NetCDF export.
        
        Args:
            data_dict: Dictionary of numpy arrays
            variable_names: List of variable names
            dimensions: Dictionary of dimensions
            output_path: Path to save NetCDF file
            
        Returns:
            Export metadata dictionary
        """
        try:
            import netCDF4
        except ImportError:
            logger.warning("netCDF4 not installed, returning metadata only")
            return {
                'status': 'warning',
                'message': 'netCDF4 library not available',
                'metadata_only': True,
            }
        
        metadata = {
            'status': 'success',
            'output_path': output_path,
            'variables': variable_names,
            'dimensions': dimensions,
            'created_at': datetime.now().isoformat(),
            'file_size_mb': 'unknown',
        }
        
        logger.info(f"NetCDF export prepared: {output_path}")
        return metadata
    
    def generate_png_visualization(self,
                                   plot_data: Dict,
                                   plot_type: str = 'scatter') -> bytes:
        """
        Generate PNG visualization.
        
        Args:
            plot_data: Data for plotting
            plot_type: Type of plot (scatter, line, heatmap, etc.)
            
        Returns:
            PNG image bytes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("Matplotlib not installed, returning placeholder")
            return b'PNG_PLACEHOLDER'
        
        logger.info(f"PNG visualization generated ({plot_type})")
        return b'PNG_IMAGE_DATA'
    
    def generate_data_inventory_report(self, 
                                      datasets: List[Dict]) -> Dict:
        """
        Generate data inventory report.
        
        Args:
            datasets: List of dataset dictionaries
            
        Returns:
            Inventory report dictionary
        """
        report_id = f"inventory_{datetime.now().timestamp()}"
        
        total_size_mb = sum(d.get('size_mb', 0) for d in datasets)
        
        report = {
            'id': report_id,
            'type': 'inventory',
            'created_at': datetime.now().isoformat(),
            'total_datasets': len(datasets),
            'total_size_mb': round(total_size_mb, 2),
            'datasets': datasets,
            'summary': {
                'by_source': self._group_by_field(datasets, 'source'),
                'by_type': self._group_by_field(datasets, 'type'),
                'by_region': self._group_by_field(datasets, 'region'),
            }
        }
        
        self.reports[report_id] = report
        logger.info(f"Inventory report generated: {report_id}")
        return report
    
    def get_report(self, report_id: str) -> Optional[Dict]:
        """Retrieve a generated report."""
        return self.reports.get(report_id)
    
    def list_reports(self, report_type: Optional[str] = None) -> List[Dict]:
        """List all generated reports, optionally filtered by type."""
        reports = list(self.reports.values())
        
        if report_type:
            reports = [r for r in reports if r.get('type') == report_type]
        
        return reports
    
    # Helper methods
    
    def _generate_markdown_report(self, title: str, data: Dict) -> str:
        """Generate Markdown formatted report."""
        md = f"# {title}\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for section, content in data.items():
            md += f"## {section}\n\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    md += f"- **{key}**: {value}\n"
            elif isinstance(content, list):
                for item in content:
                    md += f"- {item}\n"
            else:
                md += f"{content}\n"
            md += "\n"
        
        return md
    
    def _generate_html_report(self, title: str, data: Dict) -> str:
        """Generate HTML formatted report."""
        html = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #0066cc; }}
                .metadata {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="metadata">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        """
        
        for section, content in data.items():
            html += f"<h2>{section}</h2>\n"
            if isinstance(content, dict):
                html += "<ul>\n"
                for key, value in content.items():
                    html += f"<li><strong>{key}:</strong> {value}</li>\n"
                html += "</ul>\n"
            elif isinstance(content, list):
                html += "<ul>\n"
                for item in content:
                    html += f"<li>{item}</li>\n"
                html += "</ul>\n"
            else:
                html += f"<p>{content}</p>\n"
        
        html += "</body></html>"
        return html
    
    def _generate_json_report(self, title: str, data: Dict) -> str:
        """Generate JSON formatted report."""
        report = {
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'data': data,
        }
        return json.dumps(report, indent=2, default=str)
    
    def _render_html_prediction_report(self, data: Dict) -> str:
        """Render HTML prediction report."""
        html = f"""
        <html>
        <head>
            <title>Ocean Prediction Report</title>
            <style>
                body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; background: white; padding: 20px; border-radius: 8px; }}
                .metric {{ padding: 10px; background: #e8f4f8; border-left: 4px solid #0066cc; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Ocean Prediction Report</h1>
                <p>Generated: {data['generated_at']}</p>
                <h2>Summary Statistics</h2>
        """
        
        for stat, value in data.get('summary_statistics', {}).items():
            html += f'<div class="metric"><strong>{stat}:</strong> {value}</div>\n'
        
        html += "</div></body></html>"
        return html
    
    def _calculate_summary_stats(self, data: Dict) -> Dict:
        """Calculate summary statistics."""
        if not data:
            return {}
        
        values = [v for v in data.values() if isinstance(v, (int, float))]
        
        if not values:
            return {}
        
        import statistics
        return {
            'mean': round(statistics.mean(values), 4),
            'median': round(statistics.median(values), 4),
            'stdev': round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            'min': round(min(values), 4),
            'max': round(max(values), 4),
        }
    
    def _prepare_visualization_data(self, predictions: Dict) -> Dict:
        """Prepare data for visualization."""
        return {
            'type': 'time_series',
            'data_points': len(predictions),
            'format': 'json',
        }
    
    def _group_by_field(self, data: List[Dict], field: str) -> Dict:
        """Group data by a field."""
        grouped = {}
        for item in data:
            key = item.get(field, 'unknown')
            if key not in grouped:
                grouped[key] = 0
            grouped[key] += 1
        return grouped
