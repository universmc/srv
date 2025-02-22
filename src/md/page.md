##  Guide étape par étape pour créer un thème WordPress personnalisé avec des modèles PAGE.PHP et ARTICLE.PHP

Ce guide vous aidera à comprendre et à réaliser un thème WordPress personnalisé en utilisant des modèles PAGE.PHP et ARTICLE.PHP. Il est conçu pour les débutants et les utilisateurs intermédiaires qui souhaitent apprendre les bases du développement de thèmes WordPress.

**Prérequis:**

* Connaissance de base d'HTML, CSS et PHP.
* Un serveur web local ou hébergé avec WordPress installé.
* Un éditeur de code (ex : Sublime Text, Visual Studio Code).

**Étapes:**

1. **Créer un nouveau thème WordPress:**

   * Accédez à votre tableau de bord WordPress et rendez-vous dans "Apparence" > "Thèmes".
   * Cliquez sur "Ajouter un thème" puis sur "Télécharger un thème".
   * Choisissez le fichier .zip de votre thème (si vous en avez un) ou créez un nouveau thème à partir de zéro en utilisant un nom unique.
   * Activez votre nouveau thème.

2. **Créer les fichiers PAGE.PHP et ARTICLE.PHP:**

   * Dans le dossier de votre thème, créez deux nouveaux fichiers texte nommés `page.php` et `article.php`.

3. **Structurer le modèle PAGE.PHP:**

   *  Ouvrez le fichier `page.php` et ajoutez le code suivant :

     ```php
     <?php
     /**
      * Template Name: Page Template
      */
     ?>

     <!DOCTYPE html>
     <html <?php language_attributes(); ?>>
     <head>
     <meta charset="<?php bloginfo( 'charset' ); ?>">
     <meta name="viewport" content="width=device-width, initial-scale=1">
     <?php wp_head(); ?>
     </head>
     <body <?php body_class(); ?>>
     <header>
     <?php get_header(); ?>
     </header>
     <main>
     <?php
     if ( have_posts() ) :
         while ( have_posts() ) : the_post();
             the_content();
         endwhile;
     endif;
     ?>
     </main>
     <footer>
     <?php get_footer(); ?>
     </footer>
     <?php wp_footer(); ?>
     </body>
     </html>
     ```

4. **Structurer le modèle ARTICLE.PHP:**

   * Ouvrez le fichier `article.php` et ajoutez le code suivant :

     ```php
     <?php
     /**
      * Template Name: Article Template
      */
     ?>

     <!DOCTYPE html>
     <html <?php language_attributes(); ?>>
     <head>
     <meta charset="<?php bloginfo( 'charset' ); ?>">
     <meta name="viewport" content="width=device-width, initial-scale=1">
     <?php wp_head(); ?>
     </head>
     <body <?php body_class(); ?>>
     <header>
     <?php get_header(); ?>
     </header>
     <main>
     <?php
     if ( have_posts() ) :
         while ( have_posts() ) : the_post();
             ?>
             <article>
                 <h2><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
                 <?php the_excerpt(); ?>
             </article>
             <?php
         endwhile;
     endif;
     ?>
     </main>
     <footer>
     <?php get_footer(); ?>
     </footer>
     <?php wp_footer(); ?>
     </body>
     </html>
     ```

5. **Personnaliser l'apparence:**

   * Créez un fichier `style.css` dans le dossier de votre thème et ajoutez vos règles CSS pour personnaliser l'apparence de votre thème.
   * Vous pouvez utiliser les fonctions de WordPress pour accéder aux éléments de votre thème et les styliser.

6. **Tester votre thème:**

   * Créez une nouvelle page et un nouvel article pour tester vos modèles PAGE.PHP et ARTICLE.PHP.
   * Assurez-vous que le contenu s'affiche correctement et que le style est appliqué comme prévu.

**Conseils:**

* Utilisez des commentaires dans votre code pour expliquer ce que chaque partie fait.
* Développez votre thème en petites étapes et testez fréquemment.
* Consultez la documentation WordPress pour en savoir plus sur les fonctions et les filtres disponibles.
* Explorez les thèmes existants sur WordPress.org pour vous inspirer.


**Ressources supplémentaires:**

* [Documentation officielle WordPress](https://developer.wordpress.org/themes/): https://developer.wordpress.org/themes/
* [Tutoriels WordPress](https://wordpress.org/support/article/themes/): https://wordpress.org/support/article/themes/
* [WordPress Codex](https://codex.wordpress.org/): https://codex.wordpress.org/



