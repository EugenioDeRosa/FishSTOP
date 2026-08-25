# FishSTOP download site

This repository hosts the FishSTOP download page. It reads the latest release
from the desktop application repository and offers the matching macOS and
Windows installers.

- Desktop app: https://github.com/EugenioDeRosa/fishstop-desktop-email-security
- Historic Streamlit app: [`legacy-streamlit`](https://github.com/EugenioDeRosa/FishSTOP/tree/legacy-streamlit)

## Publish the website

The workflow in `.github/workflows/deploy-pages.yml` publishes `site/` to
GitHub Pages whenever `main` changes. In the repository's **Settings → Pages**,
choose **GitHub Actions** as the source if it is not already enabled.

## Publish installers

Create a GitHub Release in the desktop app repository and attach a `.dmg` for
macOS and an `.msi` or `.exe` for Windows. The page finds those files
automatically, so later releases require no site edit.

## Local preview

Open `site/index.html` in a browser, or serve the `site/` folder with any
static file server.
