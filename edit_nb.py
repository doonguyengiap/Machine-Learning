
import json
import os

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

def edit_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Refactor Logistic Regression
            if 'lr_pipeline = Pipeline(' in source and 'lr_pipeline.fit(X_train, y_train)' in source:
                new_source = source.replace(
                    '# Huấn luyện\nt0 = time.time()\nlr_pipeline.fit(X_train, y_train)\nt1 = time.time()\n\n# Dự đoán\ny_pred = lr_pipeline.predict(X_test)\n\nprint(f"Training time: {t1 - t0:.2f}s")\nprint(classification_report(y_test, y_pred))\nprint("Accuracy:", accuracy_score(y_test, y_pred))\nprint("F1:", f1_score(y_test, y_pred))\ntry:\n    print("ROC AUC:", roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1]))\nexcept Exception as e:\n    print("ROC AUC unavailable:", e)',
                    '# Huấn luyện và Cross-Validation\nlr_cv_results = cross_validate(\n    lr_pipeline,\n    X,\n    y,\n    cv=cv,\n    scoring=scoring,\n    n_jobs=-1\n)\n\nlr_results = {\n    \'Model\': \'Logistic Regression\',\n    \'Accuracy\': lr_cv_results[\'test_accuracy\'].mean(),\n    \'F1-score\': lr_cv_results[\'test_f1\'].mean(),\n    \'ROC-AUC\': lr_cv_results[\'test_roc_auc\'].mean()\n}\n\nprint(f"Logistic Regression Results: {lr_results}")'
                )
                # Handle variants with different line endings or slight spacing
                if new_source == source:
                     # Attempt a more robust replacement if plain replace failed due to formatting
                     lines = cell['source']
                     start_idx = -1
                     for i, line in enumerate(lines):
                         if '# Huấn luyện' in line:
                             start_idx = i
                             break
                     if start_idx != -1:
                         cell['source'] = lines[:start_idx] + [
                             '# Huấn luyện và Cross-Validation\n',
                             'lr_cv_results = cross_validate(\n',
                             '    lr_pipeline,\n',
                             '    X,\n',
                             '    y,\n',
                             '    cv=cv,\n',
                             '    scoring=scoring,\n',
                             '    n_jobs=-1\n',
                             ')\n',
                             '\n',
                             'lr_results = {\n',
                             "    'Model': 'Logistic Regression',\n",
                             "    'Accuracy': lr_cv_results['test_accuracy'].mean(),\n",
                             "    'F1-score': lr_cv_results['test_f1'].mean(),\n",
                             "    'ROC-AUC': lr_cv_results['test_roc_auc'].mean()\n",
                             '}\n',
                             '\n',
                             'print(f"Logistic Regression Results: {lr_results}")\n'
                         ]
                else:
                    cell['source'] = [l + '\n' for l in new_source.split('\n')]

            # Refactor KNN
            if 'knn_pipeline = Pipeline([' in source and 'search.fit(X_train, y_train)' in source:
                lines = cell['source']
                start_idx = -1
                for i, line in enumerate(lines):
                    if 't0 = time.time()' in line:
                        start_idx = i
                        break
                if start_idx != -1:
                    # Keep everything until param_grid and search definition
                    cell['source'] = lines[:start_idx] + [
                        '# Huấn luyện và Cross-Validation (sử dụng best_estimator từ GridSearchCV)\n',
                        'search.fit(X_train, y_train)\n',
                        'best_knn = search.best_estimator_\n',
                        '\n',
                        'knn_cv_results = cross_validate(\n',
                        '    best_knn,\n',
                        '    X,\n',
                        '    y,\n',
                        '    cv=cv,\n',
                        '    scoring=scoring,\n',
                        '    n_jobs=-1\n',
                        ')\n',
                        '\n',
                        'knn_results = {\n',
                        "    'Model': 'KNN',\n",
                        "    'Accuracy': knn_cv_results['test_accuracy'].mean(),\n",
                        "    'F1-score': knn_cv_results['test_f1'].mean(),\n",
                        "    'ROC-AUC': knn_cv_results['test_roc_auc'].mean()\n",
                        '}\n',
                        '\n',
                        'print(f"KNN Results: {knn_results}")\n'
                    ]

            # Update results_df
            if 'results_df = pd.DataFrame([rf_results, xgb_results])' in source:
                cell['source'] = [
                    '# Thêm SVM và Gradient Boosting\n',
                    'from sklearn.svm import SVC\n',
                    'from sklearn.ensemble import GradientBoostingClassifier\n',
                    '\n',
                    '# 1. SVM\n',
                    'svm_pipeline = Pipeline([\n',
                    "    ('preprocess', preprocessor),\n",
                    "    ('model', SVC(probability=True, random_state=42))\n",
                    '])\n',
                    'svm_cv_results = cross_validate(svm_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)\n',
                    'svm_results = {\n',
                    "    'Model': 'SVM',\n",
                    "    'Accuracy': svm_cv_results['test_accuracy'].mean(),\n",
                    "    'F1-score': svm_cv_results['test_f1'].mean(),\n",
                    "    'ROC-AUC': svm_cv_results['test_roc_auc'].mean()\n",
                    '}\n',
                    '\n',
                    '# 2. Gradient Boosting\n',
                    'gb_pipeline = Pipeline([\n',
                    "    ('preprocess', preprocessor),\n",
                    "    ('model', GradientBoostingClassifier(random_state=42))\n",
                    '])\n',
                    'gb_cv_results = cross_validate(gb_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)\n',
                    'gb_results = {\n',
                    "    'Model': 'Gradient Boosting',\n",
                    "    'Accuracy': gb_cv_results['test_accuracy'].mean(),\n",
                    "    'F1-score': gb_cv_results['test_f1'].mean(),\n",
                    "    'ROC-AUC': gb_cv_results['test_roc_auc'].mean()\n",
                    '}\n',
                    '\n',
                    '# Consolidated results\n',
                    'results_df = pd.DataFrame([rf_results, xgb_results, lr_results, knn_results, svm_results, gb_results])\n',
                    'results_df\n'
                ]

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    edit_notebook()
    print("Notebook updated successfully.")
