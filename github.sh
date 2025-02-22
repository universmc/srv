#!/bin/bash

# Variables globales
ROOT_DIR="/Volumes/devOps/apt/github"  # Chemin vers vos répertoires GitHub
GITHUB_URL="https://github.com/universmc/"

# Fonctions

# Lister les répertoires (asynchrone)
list_repos() {
  find "$ROOT_DIR" -maxdepth 1 -type d | while read dir; do
    echo "$dir" &  # Exécuter la commande en arrière-plan
  done
  wait  # Attendre la fin de toutes les commandes en arrière-plan
}

# Aller dans un répertoire (asynchrone)
goto_repo() {
  read -p "Chemin vers le répertoire (relatif à $ROOT_DIR): " path
  cd "$ROOT_DIR/$path" || echo "Répertoire '$ROOT_DIR/$path' non trouvé."
  pwd # Afficher le chemin complet du répertoire courant
}

# Lister les sous-répertoires (asynchrone)
list_subdirs() {
  find . -maxdepth 1 -type d | while read dir; do
    echo "$dir" &  # Exécuter la commande en arrière-plan
  done
  wait  # Attendre la fin de toutes les commandes en arrière-plan
}

# Mettre à jour le répertoire courant (asynchrone)
update_repo() {
  repo=$(basename "$PWD") # Récupérer le nom du répertoire courant
  git pull origin main &  # Exécuter la commande en arrière-plan
}

# Afficher l'URL GitHub du répertoire courant (asynchrone)
show_repo_url() {
  repo=$(basename "$PWD") # Récupérer le nom du répertoire courant
  echo "$GITHUB_URL$repo" &  # Exécuter la commande en arrière-plan
}

# Menu principal
while true; do
  echo "Menu:"
  echo "1. Lister les répertoires"
  echo "2. Aller dans un répertoire"
  echo "3. Lister les sous-répertoires"
  echo "4. Mettre à jour le répertoire courant"
  echo "5. Afficher l'URL GitHub du répertoire courant"
  echo "6. Quitter"

  read -p "Votre choix: " choix

  case $choix in
    1) list_repos ;;
    2) goto_repo ;;
    3) list_subdirs ;;
    4) update_repo ;;
    5) show_repo_url ;;
    6) break ;;
    *) echo "Choix invalide." ;;
  esac
done