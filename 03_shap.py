from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values_local = explainer.shap_values(X_test.iloc[0:1])

if isinstance(shap_values_local, list):
    shap_val = shap_values_local[1]
    expected_val = explainer.expected_value[1]
else:
    shap_val = shap_values_local
    expected_val = explainer.expected_value

shap.initjs()
force_plot = shap.force_plot(expected_val, shap_val, X_test.iloc[0:1], matplotlib=False)
shap.save_html("shap_force_plot.html", force_plot)

shap_values_global = explainer.shap_values(X_test)
if isinstance(shap_values_global, list):
    shap_val_global = shap_values_global[1]
else:
    shap_val_global = shap_values_global

plt.figure()
shap.summary_plot(shap_val_global, X_test, show=False)
plt.savefig('shap_summary_plot.png')
