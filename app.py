# Code adapted from Code Institute's Malaria walkthrough project

from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_leaf_visualizer import page_leaf_visualizer_body
from app_pages.page_powdery_mildew_detector import page_powdery_mildew_detector_body  # noqa
from app_pages.page_project_hypotheses import page_project_hypotheses_body
# from app_pages.page_ml_performance import page_ml_performance_metrics

# Create an instance of the app
app = MultiPage(app_name="Powdery Mildew Detector")

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Leaf Visualiser", page_leaf_visualizer_body)
app.add_page("Powdery Mildew Detection", page_powdery_mildew_detector_body)
app.add_page("Project Hypotheses", page_project_hypotheses_body)
# app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app
