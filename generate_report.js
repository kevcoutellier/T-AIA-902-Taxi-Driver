const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat
} = require("docx");

const RESULTS = "C:/Users/kevco/Documents/EPITECH/T-AIA-902-Taxi-Driver/results";

function img(filename, widthInches, heightInches) {
  const filePath = path.join(RESULTS, filename);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 200 },
    children: [new ImageRun({
      type: filename.endsWith(".png") ? "png" : "jpg",
      data: fs.readFileSync(filePath),
      transformation: { width: widthInches * 96, height: heightInches * 96 },
      altText: { title: filename, description: filename, name: filename }
    })]
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [new TextRun({ text, italics: true, size: 20, color: "666666", font: "Arial" })]
  });
}

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1B4F72", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, size: 20, color: "FFFFFF", font: "Arial" })] })]
  });
}

function dataCell(text, width, highlight) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: highlight ? { fill: "D5F5E3", type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, size: 20, font: "Arial" })] })]
  });
}

function makeTable(headers, rows, colWidths, highlightRow) {
  const tableWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map((row, ri) => new TableRow({
        children: row.map((cell, ci) => dataCell(cell, colWidths[ci], ri === highlightRow))
      }))
    ]
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, font: "Arial" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, font: "Arial" })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, font: "Arial" })]
  });
}

function para(textOrRuns, opts = {}) {
  const children = typeof textOrRuns === "string"
    ? [new TextRun({ text: textOrRuns, size: 22, font: "Arial" })]
    : textOrRuns;
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after || 160 },
    ...opts.extra,
    children
  });
}

function bold(text) { return new TextRun({ text, bold: true, size: 22, font: "Arial" }); }
function normal(text) { return new TextRun({ text, size: 22, font: "Arial" }); }
function italic(text) { return new TextRun({ text, italics: true, size: 22, font: "Arial" }); }

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1B4F72" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u00B7", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [
    // ==================== PAGE DE GARDE ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      children: [
        new Paragraph({ spacing: { before: 3000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "EPITECH", size: 32, bold: true, font: "Arial", color: "1B4F72" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "Module T-AIA-902", size: 24, font: "Arial", color: "666666" })]
        }),
        new Paragraph({ spacing: { before: 1200 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          children: [new TextRun({ text: "TAXI DRIVER", size: 72, bold: true, font: "Arial", color: "1B4F72" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Reinforcement Learning sur Taxi-v3", size: 32, font: "Arial", color: "2E75B6" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 12 } },
          children: []
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [new TextRun({ text: "Etude comparative d'algorithmes d'apprentissage par renforcement", size: 24, italics: true, font: "Arial", color: "444444" })]
        }),
        new Paragraph({ spacing: { before: 2000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2025 - 2026", size: 24, font: "Arial", color: "666666" })]
        }),
      ]
    },
    // ==================== CONTENU PRINCIPAL ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
            children: [new TextRun({ text: "Taxi Driver - Reinforcement Learning", italics: true, size: 18, color: "999999", font: "Arial" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
            children: [
              new TextRun({ text: "EPITECH - T-AIA-902  |  Page ", size: 18, color: "999999", font: "Arial" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Arial" })
            ]
          })]
        })
      },
      children: [
        // ==================== 1. INTRODUCTION ====================
        heading1("1. Introduction"),

        para("Ce rapport documente la resolution du jeu Taxi-v3 de la librairie Gymnasium par des algorithmes d'apprentissage par renforcement (Reinforcement Learning). L'objectif est de trouver l'algorithme optimal, de justifier chaque choix par des benchmarks rigoureux, et de comparer les performances de differentes approches."),

        heading2("1.1 L'environnement Taxi-v3"),

        para("Taxi-v3 est un environnement de la librairie Gymnasium qui simule un taxi dans une grille 5x5. Le taxi doit prendre un passager a un emplacement aleatoire et le deposer a une destination specifique."),

        heading3("Espace d'etats"),
        para([
          normal("L'environnement possede "),
          bold("500 etats distincts"),
          normal(", calcules comme le produit de : 25 positions du taxi (grille 5x5) x 5 positions du passager (R, G, Y, B, ou dans le taxi) x 4 destinations possibles (R, G, Y, B).")
        ]),

        heading3("Actions"),
        para("L'agent dispose de 6 actions : South (0), North (1), East (2), West (3), Pickup (4), Dropoff (5)."),

        heading3("Systeme de recompenses"),
        makeTable(
          ["Situation", "Reward"],
          [
            ["Chaque pas effectue", "-1"],
            ["Pickup ou Dropoff illegal", "-10"],
            ["Dropoff au bon endroit", "+20"],
          ],
          [5000, 4000]
        ),
        new Paragraph({ spacing: { after: 200 } }),
        para("Le systeme penalise chaque action (-1), ce qui pousse l'agent a trouver le chemin le plus court. Un pickup/dropoff mal place coute cher (-10), tandis que le succes est recompense (+20)."),

        // ==================== 2. BASELINE ====================
        heading1("2. Baseline - Algorithme Brute-Force"),

        para("L'algorithme brute-force constitue notre point de reference. Il choisit des actions completement aleatoires a chaque pas, sans aucun apprentissage. C'est l'equivalent d'un groupe controle en methode scientifique : tout algorithme RL doit faire mieux pour etre considere utile."),

        heading2("2.1 Resultats"),
        makeTable(
          ["Metrique", "Valeur"],
          [
            ["Mean Steps", "196.8"],
            ["Mean Reward", "-786.5"],
            ["Ecart-type Steps", "~30"],
            ["Ecart-type Reward", "~200"],
          ],
          [5000, 4000]
        ),
        new Paragraph({ spacing: { after: 200 } }),
        para("Avec pres de 200 pas en moyenne et un reward tres negatif, le brute-force illustre a quel point le probleme est difficile sans apprentissage. Le taxi tourne en rond, tente des pickup/dropoff illegaux, et met un temps considerable a terminer un episode."),

        // ==================== 3. Q-LEARNING ====================
        heading1("3. Q-Learning - Algorithme principal"),

        heading2("3.1 Principe"),
        para("Q-Learning est un algorithme off-policy, model-free, base sur le Temporal Difference (TD) learning. Il maintient une Q-table de taille 500 x 6 (etats x actions) initialisee a zero. Chaque cellule Q(s, a) estime le reward total espere si l'on effectue l'action a dans l'etat s, puis que l'on agit de maniere optimale par la suite."),

        heading3("Formule de mise a jour (Bellman)"),
        para([
          normal("A chaque pas, la Q-table est mise a jour selon : "),
          bold("Q(s, a) = Q(s, a) + a x [r + g x max Q(s', a') - Q(s, a)]"),
        ]),
        para([
          normal("Ou "),
          bold("a (alpha)"),
          normal(" est le learning rate, "),
          bold("g (gamma)"),
          normal(" le discount factor, "),
          bold("r"),
          normal(" la recompense immediate, et "),
          bold("max Q(s', a')"),
          normal(" la meilleure valeur Q dans l'etat suivant.")
        ]),

        heading3("Strategie epsilon-greedy"),
        para("L'agent utilise une strategie epsilon-greedy pour equilibrer exploration et exploitation. Avec une probabilite epsilon, il choisit une action aleatoire (exploration). Sinon, il choisit l'action avec la meilleure valeur Q (exploitation). Epsilon diminue progressivement au cours de l'entrainement (epsilon decay), passant d'une exploration a 100% vers une exploitation quasi-totale."),

        // ---- 3.2 Premier resultat ----
        heading2("3.2 Premier resultat non optimise"),
        para([
          normal("Pour notre premiere tentative, nous utilisons des parametres par defaut : "),
          bold("alpha=0.1, gamma=0.9, epsilon_decay=0.99"),
          normal(", avec 5000 episodes d'entrainement.")
        ]),

        makeTable(
          ["Metrique", "Brute-Force", "Q-Learning (naif)"],
          [
            ["Mean Steps", "196.8", "13.4"],
            ["Mean Reward", "-786.5", "7.6"],
            ["Amelioration", "-", "~15x"],
          ],
          [3000, 3000, 3000]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        para("Meme sans optimisation, Q-Learning est deja 15 fois plus rapide que le brute-force. La courbe d'entrainement ci-dessous montre la convergence :"),

        img("training_naive.png", 6.5, 5),
        caption("Figure 1 : Courbes d'entrainement Q-Learning non optimise (5000 episodes)"),

        para("On observe que les rewards augmentent et les steps diminuent rapidement au cours des 1500 premiers episodes. L'epsilon diminue avec le decay de 0.99, atteignant son minimum vers l'episode 500."),

        // ---- 3.3 Grid Search ----
        heading2("3.3 Optimisation par Grid Search"),
        para("Pour trouver les parametres optimaux, nous appliquons la methode scientifique : faire varier un seul parametre a la fois en fixant les autres, mesurer l'impact sur les performances, et retenir la meilleure valeur avant de passer au parametre suivant."),

        // Alpha
        heading3("Impact d'alpha (learning rate)"),
        para("Alpha controle la vitesse d'apprentissage. Un alpha trop bas empeche la convergence ; un alpha trop haut rend l'apprentissage instable."),

        makeTable(
          ["Alpha", "Mean Steps", "Mean Reward"],
          [
            ["0.01", "163.8", "-446.2"],
            ["0.1", "13.2", "7.8"],
            ["0.3", "13.0", "8.0"],
            ["0.5", "12.7", "8.3"],
            ["0.7", "13.2", "7.8"],
            ["0.9", "13.0", "8.0"],
          ],
          [3000, 3000, 3000],
          3 // highlight row 3 (alpha=0.5)
        ),
        new Paragraph({ spacing: { after: 200 } }),

        img("grid_search_alpha.png", 6.5, 2.5),
        caption("Figure 2 : Impact d'alpha sur les performances"),

        para([
          bold("Analyse : "),
          normal("alpha=0.01 est catastrophique (163.8 steps) car l'agent n'a pas le temps de converger en 5000 episodes. A partir de 0.1, les performances sont stables autour de 13 steps. Le sweet spot est "),
          bold("alpha=0.5"),
          normal(" avec 12.7 steps et un reward de 8.3.")
        ]),

        // Gamma
        heading3("Impact de gamma (discount factor)"),
        para("Gamma determine l'importance accordee aux rewards futurs. Un gamma bas rend l'agent myope ; un gamma haut le fait planifier sur le long terme."),

        makeTable(
          ["Gamma", "Mean Steps", "Mean Reward"],
          [
            ["0.5", "12.9", "8.1"],
            ["0.7", "13.4", "7.6"],
            ["0.85", "13.1", "7.9"],
            ["0.95", "12.8", "8.2"],
            ["0.99", "13.4", "7.6"],
          ],
          [3000, 3000, 3000],
          3 // highlight gamma=0.95
        ),
        new Paragraph({ spacing: { after: 200 } }),

        img("grid_search_gamma.png", 6.5, 2.5),
        caption("Figure 3 : Impact de gamma sur les performances"),

        para([
          bold("Analyse : "),
          normal("Les differences sont faibles (12.8 a 13.4 steps), ce qui indique que gamma a peu d'impact sur Taxi-v3 une fois qu'alpha est bien regle. "),
          bold("Gamma=0.95"),
          normal(" est marginalement meilleur avec 12.8 steps.")
        ]),

        // Epsilon decay
        heading3("Impact d'epsilon_decay"),
        para("Epsilon decay controle la vitesse de transition de l'exploration vers l'exploitation. Un decay rapide (0.99) fait converger epsilon vite ; un decay lent (0.999) maintient l'exploration plus longtemps."),

        makeTable(
          ["Epsilon Decay", "Mean Steps", "Mean Reward"],
          [
            ["0.990", "13.2", "7.8"],
            ["0.993", "13.6", "7.4"],
            ["0.995", "13.5", "7.5"],
            ["0.997", "13.2", "7.8"],
            ["0.999", "13.0", "8.0"],
          ],
          [3000, 3000, 3000],
          4 // highlight 0.999
        ),
        new Paragraph({ spacing: { after: 200 } }),

        img("grid_search_epsilon_decay.png", 6.5, 2.5),
        caption("Figure 4 : Impact d'epsilon_decay sur les performances"),

        para([
          bold("Analyse : "),
          normal("Un decay lent ("),
          bold("0.999"),
          normal(") donne les meilleurs resultats (13.0 steps, reward 8.0). L'exploration prolongee permet a l'agent de mieux remplir sa Q-table avant de passer en mode exploitation.")
        ]),

        // ---- 3.4 Resultat final ----
        heading2("3.4 Resultat optimise final"),

        para("En combinant les meilleurs parametres trouves par le grid search :"),

        makeTable(
          ["Parametre", "Valeur optimale"],
          [
            ["Alpha (learning rate)", "0.5"],
            ["Gamma (discount factor)", "0.95"],
            ["Epsilon decay", "0.999"],
            ["Epsilon min", "0.01"],
            ["Episodes d'entrainement", "5000"],
          ],
          [5000, 4000]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        para([
          normal("Resultat : "),
          bold("Mean steps = 13.1, Mean reward = 7.9"),
          normal(". L'agent resout le jeu en moyenne en 13 pas, contre 197 pour le brute-force, soit une amelioration de 15x.")
        ]),

        img("training_optimized.png", 6.5, 5),
        caption("Figure 5 : Courbes d'entrainement Q-Learning optimise"),

        para("La convergence est atteinte autour de l'episode 1000. On note que le decay plus lent (0.999) maintient l'exploration plus longtemps que dans la version naive, ce qui se traduit par un epsilon qui ne descend a son minimum que vers l'episode 5000."),

        // ==================== 4. MONTE CARLO ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("4. Monte Carlo - Algorithme de comparaison"),

        heading2("4.1 Principe"),
        para("Monte Carlo First-Visit est un algorithme on-policy, model-free et episodique. Contrairement a Q-Learning qui met a jour la Q-table a chaque pas (TD learning), Monte Carlo attend la fin complete de l'episode pour mettre a jour les valeurs Q."),

        para([
          bold("Fonctionnement : "),
          normal("1) Jouer un episode complet en enregistrant chaque transition (etat, action, reward). 2) Une fois l'episode termine, calculer le retour cumule G en remontant depuis la fin. 3) Pour chaque paire (etat, action) visitee pour la premiere fois dans l'episode, mettre a jour Q(s, a) avec la moyenne incrementale de tous les retours observes.")
        ]),

        para([
          bold("Difference fondamentale avec Q-Learning : "),
          normal("Monte Carlo n'utilise pas de learning rate alpha. Il calcule la moyenne exacte de tous les retours observes pour chaque paire (etat, action). Cela signifie que les premieres experiences (quand l'agent est mauvais) pesent autant que les dernieres dans la moyenne.")
        ]),

        heading2("4.2 Resultats"),

        makeTable(
          ["Configuration", "Mean Steps", "Mean Reward"],
          [
            ["Monte Carlo (5 000 ep)", "183.0", "-181.1"],
            ["Monte Carlo (50 000 ep)", "149.5", "-287.8"],
            ["Q-Learning (5 000 ep)", "13.1", "7.9"],
          ],
          [3500, 2800, 2700],
          2 // highlight Q-Learning
        ),
        new Paragraph({ spacing: { after: 200 } }),

        para("Monte Carlo ne converge pas correctement, meme avec 50 000 episodes (10x plus que Q-Learning). La courbe d'entrainement ci-dessous illustre l'echec de convergence :"),

        img("training_montecarlo_50k.png", 6.5, 5),
        caption("Figure 6 : Courbes d'entrainement Monte Carlo (50 000 episodes) - absence de convergence"),

        para("On observe que les rewards restent autour de -200 a -400 et les steps autour de 150, sans tendance claire a la baisse meme apres 50 000 episodes."),

        heading2("4.3 Analyse de l'echec"),
        para("Monte Carlo First-Visit avec moyenne incrementale echoue sur Taxi-v3 pour plusieurs raisons :"),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Poids egal des experiences : "), normal("les premieres visites (quand l'agent est mauvais) pesent autant que les dernieres dans la moyenne incrementale. Les retours initiaux tres negatifs polluent durablement les estimations Q.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Mise a jour tardive : "), normal("l'agent doit terminer un episode complet avant d'apprendre. Les episodes longs (agent debutant, ~200 pas) ralentissent enormement la boucle d'apprentissage.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Variance elevee : "), normal("les retours dans les premiers episodes ont une variance tres elevee, rendant les estimations Q instables.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [bold("Mise a jour rare : "), normal("avec 3000 paires (etat, action) possibles, chaque paire est mise a jour au plus une fois par episode, contre potentiellement plusieurs fois avec Q-Learning (TD).")]
        }),

        // ==================== 5. COMPARAISON ====================
        heading1("5. Comparaison inter-algorithmes"),

        heading2("5.1 Tableau de synthese"),

        makeTable(
          ["Algorithme", "Mean Steps", "Mean Reward", "Convergence", "Adapte"],
          [
            ["Brute-Force", "196.8", "-786.5", "N/A", "Non"],
            ["Q-Learning (naif)", "13.4", "7.6", "~1500 ep", "Oui"],
            ["Q-Learning (opt.)", "13.1", "7.9", "~1000 ep", "Optimal"],
            ["Monte Carlo (5k)", "183.0", "-181.1", "Non converge", "Non"],
            ["Monte Carlo (50k)", "149.5", "-287.8", "Non converge", "Non"],
          ],
          [2200, 1600, 1600, 1800, 1800],
          2 // highlight Q-Learning optimise
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("5.2 Comparaison visuelle"),

        img("comparison_final.png", 6.5, 2.8),
        caption("Figure 7 : Brute-Force vs Q-Learning optimise"),

        img("algo_comparison_ql_mc.png", 6.5, 4.5),
        caption("Figure 8 : Q-Learning vs Monte Carlo - barplots et courbes d'apprentissage"),

        heading2("5.3 Analyse"),
        para("Q-Learning est clairement superieur pour Taxi-v3. Trois facteurs expliquent cette dominance :"),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Temporal Difference vs Monte Carlo : "), normal("Q-Learning met a jour la Q-table a chaque pas, ce qui accelere considerablement l'apprentissage par rapport a Monte Carlo qui attend la fin de l'episode.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Learning rate alpha : "), normal("le parametre alpha permet a Q-Learning de ponderer les nouvelles experiences, donnant plus de poids aux retours recents. Monte Carlo utilise une moyenne brute, ce qui dilue l'apprentissage.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [bold("Convergence rapide : "), normal("Q-Learning converge en ~1000 episodes, la ou Monte Carlo ne converge pas meme apres 50 000 episodes sur ce probleme.")]
        }),

        // ==================== 6. ARCHITECTURE ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("6. Architecture du programme"),

        heading2("6.1 Modes d'execution"),
        para("Le programme propose trois modes :"),

        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 100 },
          children: [bold("Mode User : "), normal("l'utilisateur entre ses hyperparametres (alpha, gamma, epsilon, decay) ainsi que le nombre d'episodes d'entrainement et de test.")]
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 100 },
          children: [bold("Mode Time-limited : "), normal("utilise les parametres optimises et entraine le maximum d'episodes dans un budget de temps donne.")]
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 200 },
          children: [bold("Mode Benchmark : "), normal("lance le grid search complet automatiquement et genere tous les graphiques.")]
        }),

        heading2("6.2 Structure des fichiers"),
        makeTable(
          ["Fichier", "Role"],
          [
            ["main.py", "Point d'entree, gestion des modes via argparse"],
            ["environment.py", "Wrapper autour de Gymnasium Taxi-v3"],
            ["bruteforce.py", "Agent brute-force (baseline)"],
            ["qlearning.py", "Agent Q-Learning avec Q-table"],
            ["montecarlo.py", "Agent Monte Carlo First-Visit"],
            ["benchmark.py", "Outils de benchmark et generation de graphiques"],
          ],
          [3000, 6000]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("6.3 Utilisation"),
        para([
          normal("Exemples de commandes :\n"),
        ]),
        para([
          bold("python main.py user --alpha 0.5 --gamma 0.95 --train 5000 --test 100"),
        ]),
        para([
          bold("python main.py time --time 60 --test 100"),
        ]),
        para([
          bold("python main.py benchmark --train 5000 --test 100"),
        ]),

        // ==================== 7. CONCLUSION ====================
        heading1("7. Conclusion"),

        para("Cette etude demontre que Q-Learning tabulaire est l'algorithme optimal pour resoudre l'environnement Taxi-v3. Plusieurs enseignements cles ressortent de ce travail :"),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("L'espace d'etats conditionne le choix algorithmique : "), normal("avec seulement 500 etats discrets, une approche tabulaire est ideale. Un reseau de neurones (DQN) serait superflu et potentiellement moins performant.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Le Temporal Difference domine sur Monte Carlo : "), normal("la mise a jour a chaque pas (vs fin d'episode) et le learning rate alpha sont des avantages decisifs pour la convergence.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("L'optimisation methodique paie : "), normal("le grid search sequentiel (alpha, puis gamma, puis epsilon_decay) a permis de passer de 13.4 a 13.1 steps en moyenne, confirmant que chaque parametre contribue aux performances finales.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [bold("Le benchmark est essentiel : "), normal("sans la baseline brute-force et le comparatif Monte Carlo, il serait impossible de quantifier objectivement la qualite de notre solution Q-Learning.")]
        }),

        para([
          normal("Les parametres optimaux retenus sont : "),
          bold("alpha=0.5, gamma=0.95, epsilon_decay=0.999"),
          normal(", permettant de resoudre Taxi-v3 en "),
          bold("~13 steps"),
          normal(" en moyenne, contre ~197 pour le brute-force, soit une amelioration d'un facteur 15.")
        ]),

        new Paragraph({ spacing: { before: 400 } }),
        para([
          bold("Pistes d'amelioration : "),
          normal("pour des environnements avec des espaces d'etats plus grands ou continus, des approches comme le Deep Q-Network (DQN) ou les methodes Policy Gradient deviendraient necessaires. L'extension proposee (2 passagers, 4 destinations chacun) multiplierait l'espace d'etats et pourrait justifier l'usage de telles approches.")
        ]),
      ]
    }
  ]
});

const OUTPUT = "C:/Users/kevco/Documents/EPITECH/T-AIA-902-Taxi-Driver/rapport_taxi_driver.docx";

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("Rapport genere : " + OUTPUT);
}).catch(err => {
  console.error("Erreur:", err);
});
