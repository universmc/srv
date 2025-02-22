#!/bin/bash

# Fonction pour générer un nombre aléatoire de 75 chiffres
# Affichage de l'heure et du mode de développement
echo "💻$(date "+%H:%M:%S") /dev mode"
titre() {  
    echo "                               ╔═══════════════════════════════════════════════════════════╗";
    echo "                               ║               _                                           ║";
    echo "                               ║   _   _ _ __ (_)_   _____ _ __ ___       _ __ ___   ___   ║";
    echo "                               ║  | | | | '_ \| \ \ / / _ \ '__/ __|_____| '_ ' _' \| __|  ║";
    echo "                               ║  | |_| | | | | |\ V /  __/ |  \__ \_____| | | | | | (__   ║";
    echo "                               ║   \__,_|_| |_|_| \_/ \___|_|  |___/     |_| |_| |_|\___|  ║";
    echo "                               ║                                                           ║";
    echo "                               ╚═══════════════════════════════════════════════════════════╝";
    echo ""; 

# Définition de la fonction 'menu' pour afficher le menu avec les options:

menu() {
    echo ""
    echo "   ╔════════════════════════════════════╗    ╔════════════════════════════════════════════════════════════════════════════════════╗";
    echo "   ╠───────────{ ✨ Android  }──────────╣    ║ [📗 📕 📒]                                                               [🔎] [💫] ║";
    echo "   ║                                    ║    ╠────────────────────────────────────────────────────────────────────────────────────╣";
    echo "   ║                      💠            ║    ║                                                                                    ║";
    echo "   ║             ╲┈┈┈┈╱                 ║    ║                                                                                    ║";      
    echo "   ║             ╱▔▔▔▔╲                 ║    ║                           🔷 Weclom to android Studio 🔷                           ║";
    echo "   ║            ┃┈▇┈┈▇┈┃                ║    ║                                                                                    ║";
    echo "   ║          ╭╮┣━━━━━━┫╭╮              ║    ║                                                                                    ║";
    echo "   ║          ┃┃┃┈┈┈┈┈┈┃┃┃              ║    ║                                                                                    ║";
    echo "   ║          ╰╯┃┈┈┈┈┈┈┃╰╯              ║    ║                                                                                    ║";
    echo "   ║            ╰┓┏━━┓┏╯                ║    ║                                                                                    ║";
    echo "   ║             ╰╯  ╰╯                 ║    ║                                                                                    ║";
    echo "   ║    ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈     ║    ║                                                                                    ║";
    echo "   ║                                    ║    ║                                                                                    ║";
    echo "   ║ [>] Menu                           ║    ║                                                                                    ║";
    echo "   ║        [x] ChatBot                 ║    ║                                                                                    ║";
    echo "   ║        [ ] Generative AI           ║    ║                                                                                    ║";
    echo "   ║        [ ] MyPrompt                ║    ║                                                                                    ║";
    echo "   ║                                    ║    ║                                                                                    ║";
    echo "   ║        [ ] Google                  ║    ║                                                                                    ║";
    echo "   ║        [ ] Telegram                ║    ║                                                                                    ║";
    echo "   ║        [ ] Youtube                 ║    ║                                                                                    ║";
    echo "   ║                                    ║    ║                                                                                    ║";
    echo "   ║        [ ] gpt-wallet              ║    ║                                                                                    ║";
    echo "   ║         🔒 signal                  ║    ║                                                                                    ║";
    echo "   ║                                    ║    ║                                                                                    ║";
    echo "   ╠════════════════════════════════════╣    ╠════════════════════════════════════════════════════════════════════════════════════╣";
    echo "   ║ [📱] [📷] [🎹] [🤖] [🗂️] [📊] [💰]  ║    ║[🌴.📡]:<                                                                       [🛰] ║";
    echo "   ╚════════════════════════════════════╝    ╚════════════════════════════════════════════════════════════════════════════════════╝"; 
    echo ""

read -p "Entrez votre choix : " commande

case $commande in

        Tme)
            make Tme
            ;;
        1)
            make commande1
            ;;
        2)
            make commande2
            ;;
        3)
            make commande3
            ;;
        chat)
            make chat
            ;;
        5)
            make commande5
            ;;

        6)
            make commande6
            ;;
        7)
            make commande7
            ;;
        8)
            make commande8
            ;;
        9)
            make commande9
            ;;
        10)
            make commande10
            ;;
        a)
            make commandeA
            ;;
        g)
        make commandeg
            ;;
        r)
            clear
            menu
            ;;

        # Règle par défaut en cas d'entrée invalide
        *)
            echo "Entrée invalide"
            ;;
    esac
}
}
titre
menu  # Appel de la fonction pour afficher le menu: