#!/bin/bash

# Variables globales
ROOT_DIR="/Volumes/devOps/apt/github" # Chemin vers le répertoire contenant vos projets GitHub
GITHUB_URL="https://github.com/universmc/"

# Fonctions

# Fonction pour afficher la liste des répertoires
list_repos() {
  ls "$ROOT_DIR"
}

# Fonction pour se rendre dans un répertoire spécifique
goto_repo() {
  repo="$1"
  cd "$ROOT_DIR/$repo" || echo "Répertoire '$repo' non trouvé."
}

# Fonction pour mettre à jour un répertoire
update_repo() {
  repo="$1"
  cd "$ROOT_DIR/$repo" || echo "Répertoire '$repo' non trouvé."
  git pull origin main
}

# Fonction pour afficher l'URL GitHub d'un répertoire
show_repo_url() {
  repo="$1"
  echo "$GITHUB_URL$repo"
}

# Menu principal
while true; do
  echo "Menu:"
  echo "1. Lister les répertoires"
  echo "2. Aller dans un répertoire"
  echo "3. Mettre à jour un répertoire"
  echo "4. Afficher l'URL GitHub d'un répertoire"
  echo "5. Quitter"

  read -p "Votre choix: " choix

  case $choix in
    1) list_repos ;;
    2) read -p "Nom du répertoire: " repo; goto_repo "$repo" ;;
    3) read -p "Nom du répertoire: " repo; update_repo "$repo" ;;
    4) read -p "Nom du répertoire: " repo; show_repo_url "$repo" ;;
    5) break ;;
    *) echo "Choix invalide." ;;
  esac
done