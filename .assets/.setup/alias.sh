alias new='mkdir'
alias add='touch'
alias edit='vim'

#alias pour les listes
# Alias pour la liste des fichiers avec des options spécifiques
alias ll='ls -lh'           # Liste les fichiers avec leurs tailles humainement lisibles
alias lt='ls -lt'           # Trie les fichiers par date de modification
alias lr='ls -lR'           # Liste les fichiers récursivement

# Alias pour la liste des fichiers avec du et des options spécifiques
alias dul='du -sh *'        # Liste la taille des fichiers et des répertoires du répertoire courant
alias duh='du -sh'          # Liste la taille du répertoire courant
alias dut='du -sh * | sort -h'  # Liste la taille des fichiers et des répertoires triés par taille

# Alias pour la liste avec l'arborescence (fonctionnalité de tree)
alias tree='tree -C'        # Liste les fichiers avec des couleurs
alias treed='tree -Cd'      # Liste les fichiers et les répertoires avec des couleurs