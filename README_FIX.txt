Human Behaviour Analysis - UCI HAR fixed package

app.py = updated application with the robust UCI HAR ZIP parser.
app_old.py = backup of the previously supplied app.py.
requirements.txt = project dependencies.

Important: the original project's data/sample_har.csv, README.md, DATASET_NOTES.md,
and .streamlit configuration were not included in the files available in this chat,
so they are not reproduced here.

Run:
  py -m pip install -r requirements.txt
  py -m streamlit run app.py --server.port 8502
