"""
Data Analyst Agent for Athena

Specialized agent for data analysis, statistical processing, and
quantitative research support.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import io
import base64
import re

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..core.base_agent import BaseAgent, ResearchTask, ResearchResult


class DataAnalystAgent(BaseAgent):
    """
    Data Analyst Agent specializing in quantitative analysis and data processing.
    
    Capabilities:
    - Statistical analysis and hypothesis testing
    - Data visualization and charting
    - Trend analysis and forecasting
    - Data cleaning and preprocessing
    - Quantitative research support
    - Pattern recognition and insights
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id="data_analyst",
            name="Data Analyst Agent",
            description="Specialized agent for data analysis and quantitative research",
            capabilities=[
                "statistical_analysis",
                "data_visualization",
                "trend_analysis",
                "hypothesis_testing",
                "pattern_recognition",
                "quantitative_insights"
            ],
            config=config
        )
        
        # Initialize LLM for analysis interpretation
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        ) if openai_api_key else None
        
        # Analysis configuration
        self.max_data_points = config.get("max_data_points", 10000)
        self.visualization_format = config.get("visualization_format", "png")
        self.statistical_significance = config.get("statistical_significance", 0.05)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info("Data Analyst Agent initialized")
    
    def can_handle_task(self, task: ResearchTask) -> bool:
        """Determine if this agent can handle the given task"""
        data_indicators = [
            "data", "statistics", "analysis", "numbers", "trends",
            "metrics", "quantitative", "statistical", "correlation",
            "regression", "distribution", "average", "mean", "median",
            "chart", "graph", "visualization", "plot", "dataset"
        ]
        
        query_lower = task.query.lower()
        return any(indicator in query_lower for indicator in data_indicators)
    
    async def process_task(self, task: ResearchTask) -> ResearchResult:
        """Process a data analysis task"""
        self.logger.info(f"Processing data analysis task: {task.query}")
        
        try:
            # Determine analysis type from query
            analysis_type = self._determine_analysis_type(task.query)
            
            # Generate or simulate relevant data if not provided
            if "data" in task.context:
                data = task.context["data"]
            else:
                data = await self._generate_sample_data(task.query, analysis_type)
            
            # Perform analysis based on type
            analysis_results = await self._perform_analysis(data, analysis_type, task.query)
            
            # Create visualizations
            visualizations = await self._create_visualizations(data, analysis_type, task.query)
            
            # Generate insights and summary
            insights_summary = await self._generate_insights_summary(
                task.query, analysis_results, visualizations
            )
            
            # Calculate confidence based on data quality and analysis robustness
            confidence = self._calculate_analysis_confidence(data, analysis_results)
            
            return ResearchResult(
                task_id=task.id,
                agent_id=self.agent_id,
                content=insights_summary,
                sources=["Generated analysis", "Statistical computation"],
                confidence=confidence,
                metadata={
                    "analysis_type": analysis_type,
                    "data_points": len(data) if isinstance(data, (list, pd.DataFrame)) else 0,
                    "visualizations_created": len(visualizations),
                    "statistical_tests": list(analysis_results.keys()),
                    "processing_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in data analysis: {e}")
            raise
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis needed based on the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["trend", "time", "temporal", "over time"]):
            return "time_series"
        elif any(word in query_lower for word in ["correlation", "relationship", "association"]):
            return "correlation"
        elif any(word in query_lower for word in ["distribution", "histogram", "frequency"]):
            return "distribution"
        elif any(word in query_lower for word in ["comparison", "compare", "difference"]):
            return "comparative"
        elif any(word in query_lower for word in ["regression", "predict", "forecast"]):
            return "regression"
        else:
            return "descriptive"
    
    async def _generate_sample_data(self, query: str, analysis_type: str) -> pd.DataFrame:
        """Generate sample data relevant to the query"""
        np.random.seed(42)  # For reproducible results
        
        # Generate data based on analysis type
        if analysis_type == "time_series":
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
            values = np.cumsum(np.random.randn(len(dates))) + 100
            trend = np.linspace(0, 50, len(dates))
            data = pd.DataFrame({
                'date': dates,
                'value': values + trend,
                'category': np.random.choice(['A', 'B', 'C'], len(dates))
            })
        
        elif analysis_type == "correlation":
            n = 200
            x = np.random.randn(n)
            y = 0.7 * x + 0.3 * np.random.randn(n)
            z = np.random.randn(n)
            data = pd.DataFrame({
                'variable_x': x,
                'variable_y': y,
                'variable_z': z,
                'category': np.random.choice(['Group1', 'Group2'], n)
            })
        
        elif analysis_type == "distribution":
            n = 1000
            normal_data = np.random.normal(50, 15, n//2)
            skewed_data = np.random.exponential(2, n//2)
            data = pd.DataFrame({
                'values': np.concatenate([normal_data, skewed_data]),
                'type': ['Normal'] * (n//2) + ['Skewed'] * (n//2)
            })
        
        else:  # descriptive or comparative
            categories = ['Category A', 'Category B', 'Category C', 'Category D']
            data = pd.DataFrame({
                'category': np.random.choice(categories, 300),
                'value1': np.random.normal(100, 20, 300),
                'value2': np.random.normal(50, 10, 300),
                'group': np.random.choice(['Group 1', 'Group 2'], 300)
            })
        
        return data
    
    async def _perform_analysis(
        self,
        data: pd.DataFrame,
        analysis_type: str,
        query: str
    ) -> Dict[str, Any]:
        """Perform statistical analysis based on type"""
        results = {}
        
        try:
            if analysis_type == "time_series":
                results.update(self._time_series_analysis(data))
            elif analysis_type == "correlation":
                results.update(self._correlation_analysis(data))
            elif analysis_type == "distribution":
                results.update(self._distribution_analysis(data))
            elif analysis_type == "comparative":
                results.update(self._comparative_analysis(data))
            elif analysis_type == "regression":
                results.update(self._regression_analysis(data))
            else:
                results.update(self._descriptive_analysis(data))
                
        except Exception as e:
            self.logger.warning(f"Analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _time_series_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series analysis"""
        results = {}
        
        if 'date' in data.columns and 'value' in data.columns:
            # Basic trend analysis
            data_sorted = data.sort_values('date')
            values = data_sorted['value'].values
            
            # Calculate trend
            x = np.arange(len(values))
            trend_coef = np.polyfit(x, values, 1)[0]
            
            # Calculate volatility
            volatility = np.std(np.diff(values))
            
            # Growth rate
            growth_rate = (values[-1] - values[0]) / values[0] * 100
            
            results.update({
                "trend_coefficient": trend_coef,
                "volatility": volatility,
                "growth_rate_percent": growth_rate,
                "data_points": len(values),
                "period_start": data_sorted['date'].min().strftime('%Y-%m-%d'),
                "period_end": data_sorted['date'].max().strftime('%Y-%m-%d')
            })
        
        return results
    
    def _correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis"""
        results = {}
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            correlation_matrix = data[numeric_cols].corr()
            
            # Find strongest correlations
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_value = correlation_matrix.iloc[i, j]
                    correlations.append({
                        'variables': f"{numeric_cols[i]} vs {numeric_cols[j]}",
                        'correlation': corr_value,
                        'strength': self._interpret_correlation(abs(corr_value))
                    })
            
            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            results.update({
                "correlation_matrix": correlation_matrix.to_dict(),
                "top_correlations": correlations[:5],
                "variables_analyzed": list(numeric_cols)
            })
        
        return results
    
    def _distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform distribution analysis"""
        results = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = data[col].dropna()
            
            if len(values) > 0:
                results[f"{col}_stats"] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "skewness": float(self._calculate_skewness(values)),
                    "kurtosis": float(self._calculate_kurtosis(values))
                }
        
        return results
    
    def _comparative_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comparative analysis between groups"""
        results = {}
        
        # Find categorical and numeric columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Group statistics
            group_stats = data.groupby(cat_col)[num_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).to_dict('index')
            
            results.update({
                "group_comparison": group_stats,
                "compared_variable": num_col,
                "grouping_variable": cat_col
            })
        
        return results
    
    def _regression_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform regression analysis"""
        results = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Simple linear regression between first two numeric columns
            x = data[numeric_cols[0]].dropna()
            y = data[numeric_cols[1]].dropna()
            
            # Ensure same length
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]
            
            if len(x) > 1:
                # Calculate regression coefficients
                coeffs = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1] ** 2
                
                results.update({
                    "regression_slope": float(coeffs[0]),
                    "regression_intercept": float(coeffs[1]),
                    "r_squared": float(r_squared),
                    "independent_variable": numeric_cols[0],
                    "dependent_variable": numeric_cols[1]
                })
        
        return results
    
    def _descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        results = {
            "dataset_shape": data.shape,
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict()
        }
        
        # Numeric summaries
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results["numeric_summary"] = data[numeric_cols].describe().to_dict()
        
        # Categorical summaries
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                cat_summary[col] = {
                    "unique_values": int(data[col].nunique()),
                    "most_frequent": data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    "frequency_top5": data[col].value_counts().head().to_dict()
                }
            results["categorical_summary"] = cat_summary
        
        return results
    
    async def _create_visualizations(
        self,
        data: pd.DataFrame,
        analysis_type: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Create relevant visualizations"""
        visualizations = []
        
        try:
            if analysis_type == "time_series":
                viz = self._create_time_series_plot(data)
                if viz:
                    visualizations.append(viz)
            
            elif analysis_type == "correlation":
                viz = self._create_correlation_heatmap(data)
                if viz:
                    visualizations.append(viz)
            
            elif analysis_type == "distribution":
                viz = self._create_distribution_plot(data)
                if viz:
                    visualizations.append(viz)
            
            elif analysis_type == "comparative":
                viz = self._create_comparison_plot(data)
                if viz:
                    visualizations.append(viz)
            
            else:
                # Create general overview plots
                viz = self._create_overview_plots(data)
                visualizations.extend(viz)
                
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
        
        return visualizations
    
    def _create_time_series_plot(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create time series visualization"""
        if 'date' not in data.columns or 'value' not in data.columns:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            data_sorted = data.sort_values('date')
            ax.plot(data_sorted['date'], data_sorted['value'], linewidth=2)
            ax.set_title('Time Series Analysis')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "time_series",
                "title": "Time Series Analysis",
                "image_base64": image_base64,
                "description": "Time series plot showing trends over time"
            }
            
        except Exception as e:
            self.logger.warning(f"Time series plot creation failed: {e}")
            return None
    
    def _create_correlation_heatmap(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create correlation heatmap"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = numeric_data.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax)
            ax.set_title('Correlation Matrix')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "correlation_heatmap",
                "title": "Correlation Analysis",
                "image_base64": image_base64,
                "description": "Heatmap showing correlations between variables"
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation heatmap creation failed: {e}")
            return None
    
    def _create_distribution_plot(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create distribution visualization"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot histogram for first numeric column
            col = numeric_cols[0]
            ax.hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "distribution",
                "title": f"Distribution of {col}",
                "image_base64": image_base64,
                "description": f"Histogram showing the distribution of {col}"
            }
            
        except Exception as e:
            self.logger.warning(f"Distribution plot creation failed: {e}")
            return None
    
    def _create_comparison_plot(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create comparison visualization"""
        try:
            categorical_cols = data.select_dtypes(include=['object']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Create box plot
            data.boxplot(column=num_col, by=cat_col, ax=ax)
            ax.set_title(f'{num_col} by {cat_col}')
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "comparison",
                "title": f"{num_col} by {cat_col}",
                "image_base64": image_base64,
                "description": f"Box plot comparing {num_col} across {cat_col} categories"
            }
            
        except Exception as e:
            self.logger.warning(f"Comparison plot creation failed: {e}")
            return None
    
    def _create_overview_plots(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create general overview visualizations"""
        plots = []
        
        # Create a simple summary visualization
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot first numeric column
                col = numeric_cols[0]
                ax.hist(data[col].dropna(), bins=20, alpha=0.7)
                ax.set_title(f'Overview: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                plots.append({
                    "type": "overview",
                    "title": f"Data Overview: {col}",
                    "image_base64": image_base64,
                    "description": f"Overview visualization of {col}"
                })
                
        except Exception as e:
            self.logger.warning(f"Overview plot creation failed: {e}")
        
        return plots
    
    async def _generate_insights_summary(
        self,
        query: str,
        analysis_results: Dict[str, Any],
        visualizations: List[Dict[str, Any]]
    ) -> str:
        """Generate insights summary from analysis results"""
        if not self.llm:
            return self._generate_basic_insights_summary(query, analysis_results, visualizations)
        
        # Format analysis results for summary
        results_text = self._format_analysis_results(analysis_results)
        viz_descriptions = [viz.get("description", "") for viz in visualizations]
        
        summary_prompt = f"""
        Analyze the following data analysis results and provide comprehensive insights.
        
        Research Query: {query}
        
        Analysis Results:
        {results_text}
        
        Visualizations Created:
        {'; '.join(viz_descriptions)}
        
        Provide a comprehensive data analysis summary that includes:
        1. Key Statistical Findings
        2. Data Patterns and Trends
        3. Notable Insights and Observations
        4. Statistical Significance and Confidence
        5. Limitations and Considerations
        6. Actionable Recommendations
        
        Write in a clear, analytical style suitable for data-driven decision making.
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert data analyst. Provide comprehensive, insightful analysis summaries with clear explanations of statistical findings."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"Insights summary generation failed: {e}")
            return self._generate_basic_insights_summary(query, analysis_results, visualizations)
    
    def _format_analysis_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results for summary prompt"""
        formatted = []
        
        for key, value in results.items():
            if isinstance(value, dict):
                formatted.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    formatted.append(f"  {sub_key}: {sub_value}")
            else:
                formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def _generate_basic_insights_summary(
        self,
        query: str,
        analysis_results: Dict[str, Any],
        visualizations: List[Dict[str, Any]]
    ) -> str:
        """Generate basic insights summary without LLM"""
        summary_parts = [f"Data Analysis Summary for: {query}\n"]
        
        # Add key findings
        if analysis_results:
            summary_parts.append("Key Statistical Findings:")
            
            for key, value in analysis_results.items():
                if isinstance(value, (int, float)):
                    summary_parts.append(f"• {key}: {value:.3f}")
                elif isinstance(value, dict) and len(value) <= 5:
                    summary_parts.append(f"• {key}: {value}")
        
        # Add visualization info
        if visualizations:
            summary_parts.append(f"\nVisualizations Created: {len(visualizations)}")
            for viz in visualizations:
                summary_parts.append(f"• {viz.get('title', 'Untitled visualization')}")
        
        return "\n".join(summary_parts)
    
    def _calculate_analysis_confidence(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis"""
        confidence_factors = []
        
        # Data size factor
        data_size = len(data)
        size_factor = min(data_size / 100.0, 1.0)  # Higher confidence with more data
        confidence_factors.append(size_factor)
        
        # Data completeness factor
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        confidence_factors.append(completeness)
        
        # Analysis success factor
        analysis_success = 1.0 if analysis_results and "error" not in analysis_results else 0.3
        confidence_factors.append(analysis_success)
        
        # Statistical significance factor (if applicable)
        if "r_squared" in analysis_results:
            r_squared = analysis_results["r_squared"]
            significance_factor = min(r_squared * 2, 1.0)  # R-squared as proxy for significance
            confidence_factors.append(significance_factor)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    # Helper methods
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.8:
            return "Very Strong"
        elif correlation >= 0.6:
            return "Strong"
        elif correlation >= 0.4:
            return "Moderate"
        elif correlation >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        if std == 0:
            return 0.0
        
        skewness = (n / ((n-1) * (n-2))) * np.sum(((values - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        if std == 0:
            return 0.0
        
        kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((values - mean) / std) ** 4) - \
                  (3 * (n-1)**2 / ((n-2) * (n-3)))
        return kurtosis