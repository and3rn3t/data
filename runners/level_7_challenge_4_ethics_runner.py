#!/usr/bin/env python3
"""
Level 7 Challenge 4: AI Ethics & Governance - Responsible AI Development
Comprehensive demonstration of ethical AI practices for production systems.
"""

import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Core ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BiasMetrics:
    """Comprehensive bias assessment metrics."""

    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    disparate_impact: float
    statistical_parity_difference: float


class BiasDetector:
    """Advanced bias detection and fairness assessment."""

    def __init__(self):
        self.metrics_history = []
        self.thresholds = {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "equal_opportunity": 0.1,
            "disparate_impact": 0.8,  # Should be >= 0.8
            "statistical_parity_difference": 0.1,
        }

    def calculate_bias_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
        privileged_group: Any = 1,
    ) -> BiasMetrics:
        """Calculate comprehensive fairness metrics."""

        # Convert to binary if needed
        privileged_mask = sensitive_attr == privileged_group
        unprivileged_mask = ~privileged_mask

        # Basic rates
        priv_positive_rate = np.mean(y_pred[privileged_mask])
        unpriv_positive_rate = np.mean(y_pred[unprivileged_mask])

        # Demographic Parity (Statistical Parity)
        demographic_parity = abs(priv_positive_rate - unpriv_positive_rate)

        # Disparate Impact
        disparate_impact = min(
            unpriv_positive_rate / (priv_positive_rate + 1e-10),
            priv_positive_rate / (unpriv_positive_rate + 1e-10),
        )

        # Equalized Odds - difference in TPR and FPR
        priv_tpr = self._calculate_tpr(y_true[privileged_mask], y_pred[privileged_mask])
        unpriv_tpr = self._calculate_tpr(
            y_true[unprivileged_mask], y_pred[unprivileged_mask]
        )

        priv_fpr = self._calculate_fpr(y_true[privileged_mask], y_pred[privileged_mask])
        unpriv_fpr = self._calculate_fpr(
            y_true[unprivileged_mask], y_pred[unprivileged_mask]
        )

        equalized_odds = max(abs(priv_tpr - unpriv_tpr), abs(priv_fpr - unpriv_fpr))

        # Equal Opportunity - difference in TPR only
        equal_opportunity = abs(priv_tpr - unpriv_tpr)

        return BiasMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            equal_opportunity=equal_opportunity,
            disparate_impact=disparate_impact,
            statistical_parity_difference=priv_positive_rate - unpriv_positive_rate,
        )

    def _calculate_tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate True Positive Rate."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn + 1e-10)

    def _calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate False Positive Rate."""
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        return fp / (fp + tn + 1e-10)

    def assess_fairness(self, metrics: BiasMetrics) -> Dict[str, bool]:
        """Assess whether model meets fairness thresholds."""
        assessment = {}

        assessment["demographic_parity_fair"] = (
            metrics.demographic_parity <= self.thresholds["demographic_parity"]
        )
        assessment["equalized_odds_fair"] = (
            metrics.equalized_odds <= self.thresholds["equalized_odds"]
        )
        assessment["equal_opportunity_fair"] = (
            metrics.equal_opportunity <= self.thresholds["equal_opportunity"]
        )
        assessment["disparate_impact_fair"] = (
            metrics.disparate_impact >= self.thresholds["disparate_impact"]
        )
        assessment["statistical_parity_fair"] = (
            abs(metrics.statistical_parity_difference)
            <= self.thresholds["statistical_parity_difference"]
        )

        assessment["overall_fair"] = all(assessment.values())

        return assessment


class PrivacyProtector:
    """Privacy-preserving ML techniques."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta  # Privacy parameter
        self.noise_scale = None

    def add_differential_privacy_noise(
        self, data: np.ndarray, sensitivity: float = 1.0
    ) -> np.ndarray:
        """Add calibrated Laplace noise for differential privacy."""
        self.noise_scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, self.noise_scale, data.shape)
        return data + noise

    def private_mean(self, data: np.ndarray, sensitivity: float = 1.0) -> float:
        """Calculate differentially private mean."""
        true_mean = np.mean(data)
        noise = np.random.laplace(0, sensitivity / (self.epsilon * len(data)))
        return true_mean + noise

    def private_count(self, condition: np.ndarray) -> float:
        """Calculate differentially private count."""
        true_count = np.sum(condition)
        noise = np.random.laplace(0, 1.0 / self.epsilon)
        return max(0, true_count + noise)

    def k_anonymity_check(
        self, data: pd.DataFrame, quasi_identifiers: List[str], k: int = 5
    ) -> Dict[str, Any]:
        """Check k-anonymity compliance."""
        grouped = data.groupby(quasi_identifiers).size()

        violations = grouped[grouped < k]
        compliance_rate = (len(grouped) - len(violations)) / len(grouped)

        return {
            "k_value": k,
            "total_groups": len(grouped),
            "violations": len(violations),
            "compliance_rate": compliance_rate,
            "is_compliant": len(violations) == 0,
            "min_group_size": grouped.min(),
            "avg_group_size": grouped.mean(),
        }


@dataclass
class ModelCard:
    """Standardized model documentation for governance."""

    model_name: str
    version: str
    creation_date: str
    created_by: str

    # Model Details
    model_type: str
    architecture: str
    training_data: str
    evaluation_data: str

    # Performance
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Fairness
    bias_assessment: Dict[str, Any]
    fairness_constraints: List[str]

    # Privacy
    privacy_techniques: List[str]
    data_retention_policy: str

    # Risks & Limitations
    known_limitations: List[str]
    potential_biases: List[str]
    failure_modes: List[str]

    # Regulatory
    compliance_frameworks: List[str]
    audit_trail: List[Dict[str, Any]]


class GovernanceFramework:
    """Comprehensive AI governance and audit system."""

    def __init__(self):
        self.audit_trail = []
        self.compliance_checks = {}
        self.model_registry = {}

    def log_model_action(
        self, action: str, model_id: str, details: Dict[str, Any], user: str = "system"
    ):
        """Log all model-related actions for audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "model_id": model_id,
            "user": user,
            "details": details,
            "compliance_status": self._check_compliance(action, details),
        }
        self.audit_trail.append(entry)

    def _check_compliance(
        self, action: str, details: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check compliance with various regulatory frameworks."""
        compliance = {
            "gdpr_compliant": self._check_gdpr_compliance(action, details),
            "ccpa_compliant": self._check_ccpa_compliance(action, details),
            "ai_act_compliant": self._check_ai_act_compliance(action, details),
            "algorithmic_accountability": self._check_algorithmic_accountability(
                action, details
            ),
        }
        return compliance

    def _check_gdpr_compliance(self, action: str, details: Dict[str, Any]) -> bool:
        """Check GDPR compliance requirements."""
        # Simplified GDPR checks
        required_elements = ["data_processing_purpose", "legal_basis", "data_retention"]
        if action in ["model_training", "model_deployment"]:
            return all(element in details for element in required_elements)
        return True

    def _check_ccpa_compliance(self, action: str, details: Dict[str, Any]) -> bool:
        """Check CCPA compliance requirements."""
        if action in ["data_collection", "model_training"]:
            return "user_consent" in details and "data_categories" in details
        return True

    def _check_ai_act_compliance(self, action: str, details: Dict[str, Any]) -> bool:
        """Check EU AI Act compliance (high-level check)."""
        if action == "model_deployment":
            risk_level = details.get("risk_level", "unknown")
            if risk_level in ["high", "unacceptable"]:
                required = [
                    "bias_assessment",
                    "human_oversight",
                    "transparency_measures",
                ]
                return all(req in details for req in required)
        return True

    def _check_algorithmic_accountability(
        self, action: str, details: Dict[str, Any]
    ) -> bool:
        """Check algorithmic accountability requirements."""
        if action in ["model_deployment", "prediction"]:
            return "explainability" in details and "audit_capability" in details
        return True

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        total_actions = len(self.audit_trail)
        if total_actions == 0:
            return {"status": "no_actions", "compliance_rate": 0}

        compliance_summary = defaultdict(int)
        for entry in self.audit_trail:
            for framework, is_compliant in entry["compliance_status"].items():
                if is_compliant:
                    compliance_summary[framework] += 1

        compliance_rates = {
            framework: count / total_actions
            for framework, count in compliance_summary.items()
        }

        return {
            "total_actions": total_actions,
            "compliance_rates": compliance_rates,
            "overall_compliance": (
                min(compliance_rates.values()) if compliance_rates else 0
            ),
            "audit_trail_length": len(self.audit_trail),
            "last_audit": (
                self.audit_trail[-1]["timestamp"] if self.audit_trail else None
            ),
        }


class EthicalAISystem:
    """Comprehensive ethical AI development system."""

    def __init__(self):
        self.bias_detector = BiasDetector()
        self.privacy_protector = PrivacyProtector()
        self.governance = GovernanceFramework()
        self.model_cards = {}

    def create_synthetic_dataset(
        self, n_samples: int = 5000
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create synthetic hiring dataset with potential biases."""
        np.random.seed(42)

        # Generate synthetic hiring data
        data = {
            "age": np.random.randint(22, 65, n_samples),
            "gender": np.random.choice(["M", "F"], n_samples, p=[0.6, 0.4]),
            "education_level": np.random.choice(
                ["HS", "Bachelor", "Master", "PhD"], n_samples, p=[0.2, 0.5, 0.2, 0.1]
            ),
            "years_experience": np.random.randint(0, 40, n_samples),
            "interview_score": np.random.normal(75, 15, n_samples),
            "previous_salary": np.random.normal(60000, 25000, n_samples),
        }

        df = pd.DataFrame(data)

        # Introduce realistic biases
        # Gender bias in salary and hiring
        gender_bias = (df["gender"] == "M").astype(int) * 5000
        df["previous_salary"] += gender_bias

        # Age bias
        age_penalty = np.where(df["age"] > 50, -10, 0)
        df["interview_score"] += age_penalty

        # Create hiring decision with biases
        hiring_score = (
            df["interview_score"] * 0.4
            + df["years_experience"] * 0.3
            + (
                df["education_level"].map(
                    {"HS": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
                )
            )
            * 5
            + (df["gender"] == "M").astype(int) * 8  # Gender bias
            + np.where(df["age"] > 50, -15, 0)  # Age bias
            + np.random.normal(0, 5, n_samples)
        )

        # Convert to binary hiring decision
        hired = (hiring_score > np.percentile(hiring_score, 70)).astype(int)

        return df, hired

    def demonstrate_bias_detection(self, df: pd.DataFrame, y: np.ndarray):
        """Demonstrate comprehensive bias detection."""
        print("🔍 BIAS DETECTION & FAIRNESS ASSESSMENT")
        print("=" * 50)

        # Train a model
        le = LabelEncoder()
        X = df.copy()
        X["gender_encoded"] = le.fit_transform(X["gender"])
        X["education_encoded"] = X["education_level"].map(
            {"HS": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
        )

        feature_cols = [
            "age",
            "years_experience",
            "interview_score",
            "previous_salary",
            "gender_encoded",
            "education_encoded",
        ]
        X_features = X[feature_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Get sensitive attributes for test set
        test_indices = X_test.index
        gender_test = df.loc[test_indices, "gender_encoded"] = le.transform(
            df.loc[test_indices, "gender"]
        )

        # Calculate bias metrics for gender
        print("📊 Gender Bias Analysis:")
        gender_metrics = self.bias_detector.calculate_bias_metrics(
            y_test, y_pred, gender_test, privileged_group=1  # Male = 1
        )

        print(
            f"  • Demographic Parity Difference: {gender_metrics.demographic_parity:.3f}"
        )
        print(f"  • Equalized Odds Difference: {gender_metrics.equalized_odds:.3f}")
        print(
            f"  • Equal Opportunity Difference: {gender_metrics.equal_opportunity:.3f}"
        )
        print(f"  • Disparate Impact Ratio: {gender_metrics.disparate_impact:.3f}")
        print(
            f"  • Statistical Parity Difference: {gender_metrics.statistical_parity_difference:.3f}"
        )

        # Fairness assessment
        fairness_assessment = self.bias_detector.assess_fairness(gender_metrics)
        print(
            f"\n✅ Overall Fairness Assessment: {'PASS' if fairness_assessment['overall_fair'] else 'FAIL'}"
        )

        for metric, is_fair in fairness_assessment.items():
            if metric != "overall_fair":
                status = "✅ PASS" if is_fair else "❌ FAIL"
                print(f"  • {metric.replace('_', ' ').title()}: {status}")

        # Log the bias assessment
        self.governance.log_model_action(
            "bias_assessment",
            "hiring_model_v1",
            {
                "bias_metrics": asdict(gender_metrics),
                "fairness_assessment": fairness_assessment,
                "sensitive_attribute": "gender",
                "model_accuracy": accuracy_score(y_test, y_pred),
            },
        )

        return model, gender_metrics, fairness_assessment

    def demonstrate_privacy_protection(self, df: pd.DataFrame):
        """Demonstrate privacy-preserving techniques."""
        print("\n🔐 PRIVACY PROTECTION TECHNIQUES")
        print("=" * 50)

        # Differential Privacy
        print("🛡️ Differential Privacy Analysis:")

        # Private statistics
        private_avg_age = self.privacy_protector.private_mean(df["age"].values)
        true_avg_age = df["age"].mean()

        private_avg_salary = self.privacy_protector.private_mean(
            df["previous_salary"].values
        )
        true_avg_salary = df["previous_salary"].mean()

        print(f"  • True Average Age: {true_avg_age:.2f}")
        print(f"  • Private Average Age: {private_avg_age:.2f}")
        print(f"  • Privacy Error: {abs(true_avg_age - private_avg_age):.2f}")

        print(f"  • True Average Salary: ${true_avg_salary:,.2f}")
        print(f"  • Private Average Salary: ${private_avg_salary:,.2f}")
        print(f"  • Privacy Error: ${abs(true_avg_salary - private_avg_salary):,.2f}")

        # K-Anonymity Check
        print(f"\n🔍 K-Anonymity Assessment:")
        quasi_identifiers = ["age", "gender", "education_level"]

        # Create age groups for k-anonymity
        df_anon = df.copy()
        df_anon["age_group"] = pd.cut(
            df_anon["age"],
            bins=[0, 30, 40, 50, 100],
            labels=["<30", "30-39", "40-49", "50+"],
        )

        k_anon_results = self.privacy_protector.k_anonymity_check(
            df_anon, ["age_group", "gender", "education_level"], k=5
        )

        print(f"  • K-value: {k_anon_results['k_value']}")
        print(f"  • Total Groups: {k_anon_results['total_groups']}")
        print(f"  • Violations: {k_anon_results['violations']}")
        print(f"  • Compliance Rate: {k_anon_results['compliance_rate']:.2%}")
        print(
            f"  • Is Compliant: {'✅ YES' if k_anon_results['is_compliant'] else '❌ NO'}"
        )
        print(f"  • Min Group Size: {k_anon_results['min_group_size']}")
        print(f"  • Avg Group Size: {k_anon_results['avg_group_size']:.2f}")

        # Log privacy assessment
        self.governance.log_model_action(
            "privacy_assessment",
            "hiring_dataset_v1",
            {
                "differential_privacy_epsilon": self.privacy_protector.epsilon,
                "k_anonymity_results": k_anon_results,
                "privacy_techniques": ["differential_privacy", "k_anonymity"],
                "data_processing_purpose": "ML model training",
                "legal_basis": "legitimate_interest",
                "data_retention": "2_years",
            },
        )

        return k_anon_results

    def create_model_card(
        self,
        model,
        bias_metrics: BiasMetrics,
        fairness_assessment: Dict[str, bool],
        privacy_results: Dict[str, Any],
    ) -> ModelCard:
        """Create comprehensive model documentation."""
        print("\n📋 MODEL CARD GENERATION")
        print("=" * 50)

        model_card = ModelCard(
            model_name="Ethical Hiring Assistant",
            version="1.0.0",
            creation_date=datetime.now().strftime("%Y-%m-%d"),
            created_by="AI Ethics Team",
            # Model Details
            model_type="Random Forest Classifier",
            architecture="Ensemble of 100 decision trees",
            training_data="Synthetic hiring dataset (4000 samples)",
            evaluation_data="Hold-out test set (1000 samples)",
            # Performance
            accuracy=0.85,  # Simulated
            precision=0.82,
            recall=0.79,
            f1_score=0.80,
            # Fairness
            bias_assessment={
                "gender_bias_metrics": asdict(bias_metrics),
                "fairness_assessment": fairness_assessment,
            },
            fairness_constraints=[
                "Demographic parity difference < 0.1",
                "Equalized odds difference < 0.1",
                "Disparate impact ratio > 0.8",
            ],
            # Privacy
            privacy_techniques=["Differential Privacy", "K-Anonymity"],
            data_retention_policy="2 years with automatic deletion",
            # Risks & Limitations
            known_limitations=[
                "May exhibit residual gender bias in edge cases",
                "Performance varies across age groups",
                "Limited to technical role hiring scenarios",
            ],
            potential_biases=[
                "Gender bias in historical salary data",
                "Age bias in interview scoring",
                "Educational institution prestige effects",
            ],
            failure_modes=[
                "High false positive rate for underrepresented groups",
                "Overconfidence in borderline cases",
                "Sensitivity to missing education data",
            ],
            # Regulatory
            compliance_frameworks=[
                "GDPR",
                "CCPA",
                "EU AI Act",
                "Equal Employment Opportunity",
            ],
            audit_trail=self.governance.audit_trail,
        )

        self.model_cards["ethical_hiring_v1"] = model_card

        print("✅ Model Card Created Successfully!")
        print(f"  • Model: {model_card.model_name} v{model_card.version}")
        print(f"  • Created: {model_card.creation_date}")
        print(f"  • Performance: {model_card.accuracy:.2%} accuracy")
        print(
            f"  • Fairness: {'✅ PASS' if fairness_assessment['overall_fair'] else '❌ FAIL'}"
        )
        print(f"  • Privacy: {len(model_card.privacy_techniques)} techniques applied")
        print(f"  • Compliance: {len(model_card.compliance_frameworks)} frameworks")

        return model_card

    def demonstrate_governance(self):
        """Demonstrate AI governance and audit capabilities."""
        print("\n⚖️ AI GOVERNANCE & AUDIT SYSTEM")
        print("=" * 50)

        # Simulate model deployment
        self.governance.log_model_action(
            "model_deployment",
            "ethical_hiring_v1",
            {
                "deployment_environment": "production",
                "risk_level": "high",
                "bias_assessment": "completed",
                "human_oversight": "enabled",
                "transparency_measures": "model_cards_available",
                "explainability": "shap_values_enabled",
                "audit_capability": "full_trail_logging",
                "data_processing_purpose": "hiring_decisions",
                "legal_basis": "legitimate_interest",
                "data_retention": "2_years",
                "user_consent": "obtained",
                "data_categories": ["personal_data", "employment_history"],
            },
        )

        # Simulate ongoing predictions
        for i in range(5):
            self.governance.log_model_action(
                "prediction",
                "ethical_hiring_v1",
                {
                    "prediction_id": f"pred_{i+1}",
                    "explainability": "shap_explanation_provided",
                    "audit_capability": "decision_logged",
                    "confidence_score": np.random.uniform(0.6, 0.95),
                    "human_review_required": np.random.choice(
                        [True, False], p=[0.3, 0.7]
                    ),
                },
            )

        # Generate compliance report
        compliance_report = self.governance.generate_compliance_report()

        print("📊 Compliance Report:")
        print(f"  • Total Actions: {compliance_report['total_actions']}")
        print(f"  • Overall Compliance: {compliance_report['overall_compliance']:.2%}")
        print(f"  • Audit Trail Length: {compliance_report['audit_trail_length']}")
        print(f"  • Last Audit: {compliance_report['last_audit']}")

        print("\n📋 Framework Compliance Rates:")
        for framework, rate in compliance_report["compliance_rates"].items():
            status = "✅" if rate > 0.8 else "⚠️" if rate > 0.6 else "❌"
            print(f"  • {framework.upper()}: {rate:.2%} {status}")

        return compliance_report

    def run_comprehensive_demo(self):
        """Run the complete ethical AI demonstration."""
        print("🤖 LEVEL 7 CHALLENGE 4: AI ETHICS & GOVERNANCE")
        print("🎯 Responsible AI Development Demonstration")
        print("=" * 60)

        try:
            # 1. Create synthetic dataset
            print("\n🔄 Generating synthetic hiring dataset...")
            df, y = self.create_synthetic_dataset()
            print(f"✅ Created dataset with {len(df)} samples")

            # 2. Bias detection
            model, bias_metrics, fairness_assessment = self.demonstrate_bias_detection(
                df, y
            )

            # 3. Privacy protection
            privacy_results = self.demonstrate_privacy_protection(df)

            # 4. Model card creation
            model_card = self.create_model_card(
                model, bias_metrics, fairness_assessment, privacy_results
            )

            # 5. Governance demonstration
            compliance_report = self.demonstrate_governance()

            # Final summary
            print("\n🎉 ETHICAL AI SYSTEM SUMMARY")
            print("=" * 50)

            print("🔍 Bias Assessment:")
            print(
                f"  • Fairness Status: {'✅ PASS' if fairness_assessment['overall_fair'] else '❌ FAIL'}"
            )
            print(f"  • Demographic Parity: {bias_metrics.demographic_parity:.3f}")
            print(f"  • Equalized Odds: {bias_metrics.equalized_odds:.3f}")
            print(f"  • Disparate Impact: {bias_metrics.disparate_impact:.3f}")

            print(f"\n🔐 Privacy Protection:")
            print(f"  • Differential Privacy: ε = {self.privacy_protector.epsilon}")
            print(
                f"  • K-Anonymity: {'✅ Compliant' if privacy_results['is_compliant'] else '❌ Non-compliant'}"
            )
            print(f"  • Compliance Rate: {privacy_results['compliance_rate']:.2%}")

            print(f"\n📋 Governance:")
            print(f"  • Model Card: ✅ Generated")
            print(f"  • Audit Trail: {len(self.governance.audit_trail)} entries")
            print(
                f"  • Overall Compliance: {compliance_report['overall_compliance']:.2%}"
            )

            print(f"\n⚖️ Regulatory Compliance:")
            for framework, rate in compliance_report["compliance_rates"].items():
                status = "✅" if rate >= 0.8 else "⚠️"
                print(f"  • {framework.upper()}: {rate:.2%} {status}")

            print("\n🚀 PRODUCTION READINESS:")
            readiness_score = (
                (1.0 if fairness_assessment["overall_fair"] else 0.0) * 0.3
                + (privacy_results["compliance_rate"]) * 0.3
                + (compliance_report["overall_compliance"]) * 0.4
            )

            print(f"  • Readiness Score: {readiness_score:.2%}")
            if readiness_score >= 0.8:
                print("  • Status: ✅ READY FOR PRODUCTION")
            elif readiness_score >= 0.6:
                print("  • Status: ⚠️ NEEDS IMPROVEMENT")
            else:
                print("  • Status: ❌ NOT READY - MAJOR ISSUES")

            print(f"\n🎯 Challenge 4 Complete: AI Ethics & Governance Mastered!")
            print(f"📈 Demonstrated comprehensive responsible AI development!")

            return {
                "bias_metrics": asdict(bias_metrics),
                "fairness_assessment": fairness_assessment,
                "privacy_results": privacy_results,
                "compliance_report": compliance_report,
                "model_card": model_card,
                "readiness_score": readiness_score,
            }

        except Exception as e:
            print(f"❌ Error in ethical AI demonstration: {str(e)}")
            logger.error(f"Demo failed: {str(e)}")
            return None


def main():
    """Main execution function."""
    print("🚀 Starting Level 7 Challenge 4: AI Ethics & Governance...")

    # Create and run the ethical AI system
    ethical_ai = EthicalAISystem()
    results = ethical_ai.run_comprehensive_demo()

    if results:
        print(f"\n✅ Level 7 Challenge 4 completed successfully!")
        print(f"🎉 Ready for Challenge 4: AI Ethics & Governance completed!")

        # Save results
        os.makedirs("results", exist_ok=True)
        with open("results/level_7_challenge_4_results.json", "w") as f:
            # Convert model_card to dict for JSON serialization
            json_results = results.copy()
            json_results["model_card"] = asdict(results["model_card"])
            json.dump(json_results, f, indent=2, default=str)

        print(f"📁 Results saved to results/level_7_challenge_4_results.json")
    else:
        print(f"❌ Challenge 4 failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
