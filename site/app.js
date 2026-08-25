const releaseApi = "https://api.github.com/repos/EugenioDeRosa/fishstop-desktop-email-security/releases/latest";
const releasesPage = "https://github.com/EugenioDeRosa/fishstop-desktop-email-security/releases";

const findAsset = (assets, platform) => assets.find(({ name }) => {
  const filename = name.toLowerCase();
  return platform === "macos" ? filename.endsWith(".dmg") : filename.endsWith(".msi") || filename.endsWith(".exe");
});

const setupButton = (platform, asset, version) => {
  const button = document.querySelector(`[data-download="${platform}"]`);
  const versionLabel = document.querySelector(`[data-version="${platform}"]`);
  if (!asset) {
    button.href = releasesPage;
    button.textContent = "Vedi le release";
    button.append(" →");
    button.setAttribute("aria-disabled", "false");
    versionLabel.textContent = "Installer non ancora pubblicato";
    return false;
  }
  button.href = asset.browser_download_url;
  button.download = asset.name;
  button.textContent = "Scarica ora";
  button.append(" →");
  button.setAttribute("aria-disabled", "false");
  versionLabel.textContent = `${version} · ${asset.name}`;
  return true;
};

fetch(releaseApi, { headers: { Accept: "application/vnd.github+json" } })
  .then((response) => {
    if (!response.ok) throw new Error("release unavailable");
    return response.json();
  })
  .then((release) => {
    const macReady = setupButton("macos", findAsset(release.assets, "macos"), release.tag_name);
    const windowsReady = setupButton("windows", findAsset(release.assets, "windows"), release.tag_name);
    const status = document.querySelector("#release-status");
    const available = [macReady, windowsReady].filter(Boolean).length;
    status.textContent = available ? `Ultima versione: ${release.name || release.tag_name}` : "L'ultima release non contiene ancora installer.";
    status.className = `release-status ${available ? "ready" : "error"}`;
  })
  .catch(() => {
    document.querySelectorAll("[data-download]").forEach((button) => {
      button.href = releasesPage;
      button.textContent = "Vedi le release";
      button.append(" →");
      button.setAttribute("aria-disabled", "false");
    });
    document.querySelectorAll("[data-version]").forEach((label) => { label.textContent = "Controlla la pagina delle release"; });
    const status = document.querySelector("#release-status");
    status.textContent = "Impossibile verificare la release in questo momento.";
    status.className = "release-status error";
  });
