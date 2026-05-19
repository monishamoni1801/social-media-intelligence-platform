import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class TrendPredictor:
    def __init__(self, data_path='../data/processed_data.csv'):
        self.df = pd.read_csv(data_path)
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()

    def prepare_features(self):
        """Separate features and target"""
        feature_cols = ['num_hashtags', 'avg_hashtag_length', 'engagement_rate',
                        'like_retweet_ratio', 'textblob_polarity', 'vader_compound',
                        'hour_sin', 'hour_cos', 'is_weekend', 'month']
        
        X = self.df[feature_cols]
        y = self.df['will_trend']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        return X, y

    def train_models(self):
        """Train both Random Forest and XGBoost"""
        X, y = self.prepare_features()
        
        # Scale features (important for some models, though tree-based are robust)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest with hyperparameter tuning
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params, cv=5, scoring='roc_auc', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.rf_model = rf_grid.best_estimator_
        
        # XGBoost
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            xgb_params, cv=5, scoring='roc_auc', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        self.xgb_model = xgb_grid.best_estimator_
        
        # Evaluate
        self.evaluate_models(X_test, y_test)
        
        # Save models
        joblib.dump(self.rf_model, '../models/random_forest.pkl')
        joblib.dump(self.xgb_model, '../models/xgboost.pkl')
        joblib.dump(self.scaler, '../models/scaler.pkl')
        
        return self.rf_model, self.xgb_model

    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        models = {'Random Forest': self.rf_model, 'XGBoost': self.xgb_model}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            print(f"\n{'='*50}")
            print(f"{name} Performance")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred))
            print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
            
            # Cross-validation score
            X, y = self.prepare_features()
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_cols = ['num_hashtags', 'avg_hashtag_length', 'engagement_rate',
                                'like_retweet_ratio', 'textblob_polarity', 'vader_compound',
                                'hour_sin', 'hour_cos', 'is_weekend', 'month']
                importances = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 5 features for {name}:")
                print(importances.head(5))
                
                # Plot feature importance
                plt.figure(figsize=(8, 5))
                plt.barh(importances['feature'][:5], importances['importance'][:5])
                plt.xlabel('Importance')
                plt.title(f'{name} - Feature Importance')
                plt.tight_layout()
                plt.savefig(f'../models/{name.lower()}_importance.png')

if __name__ == "__main__":
    predictor = TrendPredictor()
    rf, xgb = predictor.train_models()