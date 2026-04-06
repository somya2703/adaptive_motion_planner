# Setting up your GitHub repo

## 1. Create the repo

```bash
cd adaptive_motion_planner
git init
git add .
git commit -m "feat: initial implementation — Informed RRT*, CBF safety, 7-DoF IK"
```

Go to https://github.com/new and create a repo named `adaptive_motion_planner`.

```bash
git remote add origin https://github.com/YOUR_USERNAME/adaptive_motion_planner.git
git branch -M main
git push -u origin main
```

## 2. Generate result images locally (commit them so README renders)

```bash
mkdir -p results/plans results/trajectories results/cbf
docker build -t amp .
docker run --rm -v $(pwd)/results:/app/results amp python pipeline.py --quick
git add results/
git commit -m "results: add pipeline output images"
git push
```

This makes the images in `README.md` render directly on your GitHub profile page.

## 3. Enable GitHub Actions

The CI workflow at `.github/workflows/ci.yml` runs automatically on every push.
It will:
- Build the Docker image (cached after first run)
- Run the 38-test suite and publish results in the PR view
- Run the quick pipeline and upload all plots as a workflow artifact
- Run the full 10-trial benchmark on pushes to `main`

No setup needed — it works as soon as you push to GitHub.

## 4. Add a CI badge to your README

Replace `YOUR_USERNAME` in the badge URL at the top of `README.md`:

```
[![CI](https://github.com/YOUR_USERNAME/adaptive_motion_planner/actions/workflows/ci.yml/badge.svg)](...)
```

## 5. Recommended commit sequence for a clean history

```bash
git log --oneline
# Should look something like:
# abc1234 results: add pipeline output images
# def5678 feat: initial implementation — Informed RRT*, CBF safety, 7-DoF IK
```

That's a clean, professional commit history for a portfolio project.
