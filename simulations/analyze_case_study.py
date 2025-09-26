"""
Case Study Analysis
==================

Analyze the e-commerce platform case study data and evaluate the DOS system's performance.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import seaborn as sns

class CaseStudyAnalyzer:
    """Analyze the e-commerce case study data."""
    
    def __init__(self, data_dir: str = "case_study_data"):
        self.data_dir = data_dir
        self.metrics = []
        self.incidents = []
        self.deployments = []
        
    def load_data(self):
        """Load all case study data."""
        # Load metrics
        with open(f"{self.data_dir}/metrics.jsonl") as f:
            self.metrics = [json.loads(line) for line in f]
        
        # Load incidents
        with open(f"{self.data_dir}/incidents.json") as f:
            self.incidents = json.load(f)
        
        # Load deployments
        with open(f"{self.data_dir}/deployments.json") as f:
            self.deployments = json.load(f)
        
        # Convert timestamps
        for item in self.metrics + self.incidents + self.deployments:
            if 'timestamp' in item:
                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
            if 'start_time' in item:
                item['start_time'] = datetime.fromisoformat(item['start_time'])
            if 'end_time' in item:
                item['end_time'] = datetime.fromisoformat(item['end_time'])
    
    def analyze_incidents(self):
        """Analyze incident patterns and detection performance."""
        print("\n📊 Incident Analysis")
        print("=" * 50)
        
        # Basic stats
        print(f"Total incidents: {len(self.incidents)}")
        
        # Incident types
        incident_types = pd.Series([i['type'] for i in self.incidents]).value_counts()
        print("\nIncident Types:")
        print(incident_types.to_string())
        
        # Time to detect/resolve
        ttd = [i['time_to_detect'] for i in self.incidents]
        ttr = [i['time_to_resolve'] for i in self.incidents]
        print(f"\nAverage Time to Detect: {np.mean(ttd)/60:.1f} minutes")
        print(f"Average Time to Resolve: {np.mean(ttr)/60:.1f} minutes")
        
        # Detection sources
        detection_sources = pd.Series([i['detected_by'] for i in self.incidents]).value_counts()
        print("\nDetection Sources:")
        print(detection_sources.to_string())
        
        return incident_types
    
    def analyze_deployments(self):
        """Analyze deployment patterns and stability."""
        print("\n🚀 Deployment Analysis")
        print("=" * 50)
        
        df = pd.DataFrame(self.deployments)
        
        # Deployment success rate
        success_rate = df['status'].value_counts(normalize=True) * 100
        print("\nDeployment Status (%):")
        print(success_rate.to_string())
        
        # Deployment frequency by service
        deployments_by_service = df['service'].value_counts()
        print("\nDeployments by Service:")
        print(deployments_by_service.to_string())
        
        # Deployment duration
        print(f"\nAverage deployment duration: {df['duration_seconds'].mean():.1f} seconds")
        
        return df
    
    def plot_incident_timeline(self):
        """Create a timeline visualization of incidents."""
        plt.figure(figsize=(12, 6))
        
        for i, incident in enumerate(self.incidents):
            start = incident['start_time']
            end = incident['end_time']
            severity = incident['severity']
            
            # Map severity to y-position and color
            y_pos = {'low': 1, 'medium': 2, 'high': 3}[severity]
            color = {'low': 'green', 'medium': 'orange', 'high': 'red'}[severity]
            
            plt.hlines(y_pos, start, end, colors=color, linewidth=3)
            plt.text(start, y_pos + 0.1, incident['type'], ha='left', va='bottom')
        
        plt.title('Incident Timeline by Severity')
        plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('incident_timeline.png')
        print("\n✅ Saved incident timeline to incident_timeline.png")
    
    def plot_service_metrics(self):
        """Visualize service metrics over time."""
        try:
            df = pd.DataFrame(self.metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Select only numeric columns for resampling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[['timestamp', 'service'] + list(numeric_cols)]
            
            # Resample to hourly means for each service
            hourly = df_numeric.set_index('timestamp').groupby('service').resample('1H').mean()
            
            # Plot CPU usage
            plt.figure(figsize=(12, 6))
            for service in df['service'].unique():
                try:
                    service_data = hourly.loc[service]
                    plt.plot(service_data.index, service_data['cpu_usage'], label=service)
                except KeyError:
                    continue
            
            plt.title('Hourly CPU Usage by Service')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('cpu_usage.png')
            print("✅ Saved CPU usage plot to cpu_usage.png")
        except Exception as e:
            print(f"⚠️ Error generating metrics plot: {str(e)}")
            
    def plot_incident_impact(self):
        """Analyze impact of incidents on service metrics."""
        try:
            df = pd.DataFrame(self.metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Mark time periods with incidents
            for incident in self.incidents:
                mask = (df['timestamp'] >= pd.to_datetime(incident['start_time'])) & \
                       (df['timestamp'] <= pd.to_datetime(incident['end_time']))
                df.loc[mask, 'incident_active'] = 1
                df.loc[mask, 'incident_type'] = incident['type']
            
            # Compare metrics during incidents vs normal operation
            incident_metrics = df[df['incident_active'] == 1].select_dtypes(include=[np.number]).mean()
            normal_metrics = df[df['incident_active'] != 1].select_dtypes(include=[np.number]).mean()
            
            # Create impact report
            impact = pd.DataFrame({
                'normal': normal_metrics,
                'during_incident': incident_metrics,
                'impact': (incident_metrics - normal_metrics) / normal_metrics * 100
            })
            
            print("\n📈 Incident Impact Analysis")
            print("=" * 50)
            print("\nAverage Metrics During Incidents vs Normal Operation:")
            print(impact[['normal', 'during_incident', 'impact']].round(2).to_string())
            
            # Save impact report
            impact.to_csv('incident_impact_report.csv')
            print("\n✅ Saved detailed impact report to incident_impact_report.csv")
            
        except Exception as e:
            print(f"⚠️ Error analyzing incident impact: {str(e)}")

def main():
    """Run the case study analysis."""
    analyzer = CaseStudyAnalyzer()
    
    print("🔍 Loading case study data...")
    analyzer.load_data()
    
    # Run analyses
    analyzer.analyze_incidents()
    analyzer.analyze_deployments()
    
    # Generate visualizations and reports
    print("\n📊 Generating reports and visualizations...")
    analyzer.plot_incident_timeline()
    analyzer.plot_service_metrics()
    analyzer.plot_incident_impact()
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("- incident_timeline.png: Timeline of all incidents")
    print("- cpu_usage.png: Service CPU usage patterns")
    print("- incident_impact_report.csv: Detailed impact analysis")
    
    # Print summary of findings
    print("\n📝 Key Findings:")
    print("1. Incident detection and response metrics")
    print("2. Deployment success rates and patterns")
    print("3. Service performance during incidents vs normal operation")
    print("4. Impact analysis of different incident types")

if __name__ == "__main__":
    main()
