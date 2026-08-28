"""Train/test splitting and feature scaling."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified split so both classes keep their proportions in train and test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set : {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize features to zero mean and unit variance.

    This matters a lot for SVMs. The algorithm works with distances between
    points, so a feature measured in the hundreds ('mean area') would
    completely drown out one measured in thousandths ('mean smoothness')
    if left unscaled.

    The scaler is fit on the TRAINING data only, then applied to both sets.
    Fitting on the full dataset would leak test-set statistics into training.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler