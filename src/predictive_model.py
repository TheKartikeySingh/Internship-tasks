from sklearn.ensemble import RandomForestClassifier

def train_delay_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
