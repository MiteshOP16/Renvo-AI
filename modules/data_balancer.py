import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataBalancer:
    """Handles various data balancing techniques for imbalanced datasets"""
    
    def __init__(self):
        self.balancing_methods = {
            'Random Oversampling': self._random_oversampling,
            'SMOTE': self._smote,
            'Random Undersampling': self._random_undersampling,
            'Tomek Links': self._tomek_links,
            'NearMiss-1': self._nearmiss_1,
            'NearMiss-2': self._nearmiss_2,
            'NearMiss-3': self._nearmiss_3,
            'ENN': self._enn,
            'CNN': self._cnn,
            'OSS': self._oss,
            'Cluster Centroids': self._cluster_centroids,
            'NCR': self._ncr,
            'SMOTE + Tomek Links': self._smote_tomek,
            'SMOTE + ENN': self._smote_enn,
        }
        self._label_encoder = None
        self._target_was_encoded = False
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """Return categorized list of available balancing methods"""
        return {
            'Oversampling': ['Random Oversampling', 'SMOTE'],
            'Undersampling': [
                'Random Undersampling', 'Tomek Links', 
                'NearMiss-1', 'NearMiss-2', 'NearMiss-3',
                'ENN', 'CNN', 'OSS', 
                'Cluster Centroids', 'NCR'
            ],
            'Hybrid': ['SMOTE + Tomek Links', 'SMOTE + ENN']
        }
    
    def validate_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Any]:
        """Validate data quality before balancing"""
        errors = []
        warnings = []
        
        if df is None or df.empty:
            errors.append("Dataset is empty or not loaded")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        if not feature_cols:
            errors.append("No feature columns selected")
        
        if not target_col:
            errors.append("No target column selected")
        
        if target_col and target_col not in df.columns:
            errors.append(f"Target column '{target_col}' not found in dataset")
        
        for col in feature_cols:
            if col not in df.columns:
                errors.append(f"Feature column '{col}' not found in dataset")
        
        if target_col and target_col in df.columns:
            if df[target_col].isnull().any():
                errors.append(f"Target column '{target_col}' contains missing values. Please clean this column using the Cleaning Wizard first.")
            
            unique_vals = df[target_col].nunique()
            if unique_vals < 2:
                errors.append(f"Target column must have at least 2 classes (found {unique_vals})")
            elif unique_vals > 10:
                warnings.append(f"Target column has {unique_vals} classes. Balancing works best with fewer classes.")
        
        categorical_features = []
        for col in feature_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    errors.append(f"Feature column '{col}' contains missing values. Please clean this column using the Cleaning Wizard first.")
                
                if not pd.api.types.is_numeric_dtype(df[col]):
                    categorical_features.append(col)
        
        if categorical_features:
            errors.append(f"Feature columns {categorical_features} are not numeric. Balancing requires numeric features. Please encode categorical variables using the Column Analysis page before balancing, or select only numeric columns.")
        
        if len(df) < 10:
            errors.append("Dataset has fewer than 10 rows. Need more data for balancing.")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_class_distribution(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Get the distribution of classes in the target column"""
        return df[target_col].value_counts().sort_index()
    
    def stratified_split(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Perform stratified train/test split"""
        try:
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y, 
                random_state=random_state
            )
            
            train_df = X_train.copy()
            train_df[target_col] = y_train
            
            test_df = X_test.copy()
            test_df[target_col] = y_test
            
            return {
                'success': True,
                'train_data': train_df,
                'test_data': test_df,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_distribution': y_train.value_counts().sort_index(),
                'test_distribution': y_test.value_counts().sort_index()
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error during stratified split: {str(e)}"
            }
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare feature data for sampling - ensure proper NumPy array with correct dtype"""
        X_prepared = np.asarray(X, dtype=np.float64)
        if X_prepared.ndim == 1:
            X_prepared = X_prepared.reshape(-1, 1)
        return X_prepared
    
    def _prepare_target(self, y: np.ndarray) -> Tuple[np.ndarray, Optional[LabelEncoder], bool]:
        """Prepare target data for sampling - encode if non-numeric, return encoder for inverse transform"""
        y_prepared = np.asarray(y).ravel()
        
        if not np.issubdtype(y_prepared.dtype, np.number):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_prepared)
            return y_encoded, le, True
        
        return y_prepared, None, False
    
    def _safe_fit_resample(self, sampler, X: np.ndarray, y: np.ndarray, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Safely apply fit_resample with proper error handling and label restoration"""
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            if was_encoded and label_encoder is not None:
                y_resampled = label_encoder.inverse_transform(y_resampled)
            
            return X_resampled, y_resampled
        except Exception as e:
            raise RuntimeError(f"Sampling failed: {str(e)}")
    
    def balance_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str, 
        method: str,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Apply balancing method to the data"""
        try:
            X = df[feature_cols].values
            y = df[target_col].values
            
            original_dist = self.get_class_distribution(df, target_col)
            
            if method not in self.balancing_methods:
                if method in ['GAN Oversampling', 'VAE Oversampling', 'Cost-Sensitive Learning']:
                    return {
                        'success': False,
                        'error': f"{method} is not yet implemented. This advanced method requires additional dependencies.",
                        'original_distribution': original_dist
                    }
                return {
                    'success': False,
                    'error': f"Unknown balancing method: {method}",
                    'original_distribution': original_dist
                }
            
            X_prepared = self._prepare_features(X)
            y_prepared, label_encoder, was_encoded = self._prepare_target(y)
            
            balancer_func = self.balancing_methods[method]
            X_balanced, y_balanced = balancer_func(X_prepared, y_prepared, random_state, label_encoder, was_encoded)
            
            balanced_df = pd.DataFrame(X_balanced, columns=feature_cols)
            balanced_df[target_col] = y_balanced
            
            balanced_dist = balanced_df[target_col].value_counts().sort_index()
            
            return {
                'success': True,
                'balanced_data': balanced_df,
                'original_distribution': original_dist,
                'balanced_distribution': balanced_dist,
                'method': method,
                'original_size': len(df),
                'balanced_size': len(balanced_df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error during balancing: {str(e)}",
                'original_distribution': self.get_class_distribution(df, target_col) if target_col in df.columns else pd.Series()
            }
    
    def _random_oversampling(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Random Oversampling"""
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _smote(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE (Synthetic Minority Over-sampling Technique)"""
        from imblearn.over_sampling import SMOTE
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1:
            k_neighbors = 1
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _random_undersampling(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Random Undersampling"""
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _tomek_links(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Tomek Links"""
        from imblearn.under_sampling import TomekLinks
        sampler = TomekLinks()
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _nearmiss_1(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-1"""
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(version=1)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _nearmiss_2(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-2"""
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(version=2)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _nearmiss_3(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-3"""
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(version=3)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _enn(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Edited Nearest Neighbours"""
        from imblearn.under_sampling import EditedNearestNeighbours
        sampler = EditedNearestNeighbours()
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _cnn(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Condensed Nearest Neighbour"""
        from imblearn.under_sampling import CondensedNearestNeighbour
        sampler = CondensedNearestNeighbour(random_state=random_state)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _oss(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """One-Sided Selection"""
        from imblearn.under_sampling import OneSidedSelection
        sampler = OneSidedSelection(random_state=random_state)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _cluster_centroids(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster Centroids"""
        from imblearn.under_sampling import ClusterCentroids
        sampler = ClusterCentroids(random_state=random_state)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _ncr(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Neighbourhood Cleaning Rule"""
        from imblearn.under_sampling import NeighbourhoodCleaningRule
        sampler = NeighbourhoodCleaningRule()
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _smote_tomek(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE + Tomek Links (Hybrid)"""
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1:
            k_neighbors = 1
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        sampler = SMOTETomek(random_state=random_state, smote=smote)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
    
    def _smote_enn(self, X: np.ndarray, y: np.ndarray, random_state: int, label_encoder: Optional[LabelEncoder], was_encoded: bool) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE + ENN (Hybrid)"""
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1:
            k_neighbors = 1
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        sampler = SMOTEENN(random_state=random_state, smote=smote)
        return self._safe_fit_resample(sampler, X, y, label_encoder, was_encoded)
