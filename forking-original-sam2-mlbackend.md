# Forking the original SAM2 ML backend

SAM2Plus began as a copy of HumanSignal's stock
[`segment_anything_2_image`](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)
example. These are the notes used to bootstrap the fork — kept for reference and
for anyone re-basing on a newer upstream example.

```bash
mkdir SAM2Plus && cd SAM2Plus
git init
git remote add upstream https://github.com/HumanSignal/label-studio-ml-backend.git
git fetch upstream master
git archive upstream/master label_studio_ml/examples/segment_anything_2_image | tar -x --strip-components=3
git add -A && git commit -m "init sam2 ml-backend"
```

Then update `model.py`, including renaming `class NewModel` to `class SAM2Plus`
and making the corresponding changes in `_wsgi.py`.
