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
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "·", alignment: AlignmentType.LEFT,
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
          children: [new TextRun({ text: "Analyse approfondie — Monte Carlo First-Visit", size: 32, font: "Arial", color: "2E75B6" })]
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
          children: [new TextRun({ text: "Optimisation des hyperparametres, evaluation sur 8 metriques et comparaison inter-algorithmes sur Taxi-v3", size: 24, italics: true, font: "Arial", color: "444444" })]
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
            children: [new TextRun({ text: "Taxi Driver — Monte Carlo First-Visit", italics: true, size: 18, color: "999999", font: "Arial" })]
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

        // ==================== 1. CONTEXTE ====================
        heading1("1. Contexte et environnement"),

        para("Ce rapport analyse l'algorithme Monte Carlo First-Visit applique a la resolution de l'environnement Taxi-v3 de la librairie Gymnasium. L'objectif est de justifier methodiquement les choix d'hyperparametres, d'evaluer les performances sur 8 metriques communes, et de positionner Monte Carlo par rapport aux autres approches du groupe."),

        heading2("1.1 L'environnement Taxi-v3"),
        para("Taxi-v3 simule un taxi dans une grille 5x5. Le taxi doit prendre un passager a un emplacement aleatoire parmi 4 stations (R, G, Y, B) et le deposer a la bonne destination. L'agent doit donc maitriser une sequence de comportements : se deplacer vers le passager, l'embarquer, se deplacer vers la destination, le deposer."),

        heading3("Espace d'etats et actions"),
        para([
          normal("L'environnement possede "),
          bold("500 etats discrets"),
          normal(" (25 positions taxi x 5 positions passager x 4 destinations) et "),
          bold("6 actions"),
          normal(" : South, North, East, West, Pickup, Dropoff.")
        ]),

        heading3("Systeme de recompenses"),
        makeTable(
          ["Situation", "Reward"],
          [
            ["Chaque pas effectue", "-1"],
            ["Pickup ou Dropoff illegal", "-10"],
            ["Dropoff au bon endroit (succes)", "+20"],
          ],
          [5500, 3500]
        ),
        new Paragraph({ spacing: { after: 200 } }),
        para("La structure de recompenses cree un probleme de credit assignment complexe : le signal positif (+20) est retarde de plusieurs dizaines de pas apres les premieres decisions. Cela pose un defi specifique pour Monte Carlo, comme nous le montrons dans les sections suivantes."),

        heading2("1.2 Metriques communes du groupe"),
        para("Afin de permettre des comparaisons objectives entre les algorithmes implementes par chaque membre du groupe, nous evaluons tous nos agents sur les 8 metriques suivantes, calculees sur 100 episodes de test en mode greedy :"),
        makeTable(
          ["#", "Metrique", "Definition"],
          [
            ["1", "Reward moyen", "mean(rewards) sur les episodes de test"],
            ["2", "Ecart-type reward", "std(rewards) — mesure la stabilite"],
            ["3", "Steps moyens", "mean(steps) — nombre de pas par episode"],
            ["4", "Ecart-type steps", "std(steps)"],
            ["5", "Taux de succes", "mean(reward > 0) x 100 — % d'episodes reussis"],
            ["6", "Reward / step", "sum(rewards) / sum(steps) — efficacite par action"],
            ["7", "Actes illegaux", "count(reward == -10) — Pickup/Dropoff mal places"],
            ["8", "Episode de convergence", "1er episode ou succes >= 25% sur 50 ep. consecutifs"],
          ],
          [400, 2400, 6200]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        // ==================== 2. PRINCIPE ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("2. Principe de Monte Carlo First-Visit"),

        heading2("2.1 Fonctionnement general"),
        para("Monte Carlo First-Visit est un algorithme on-policy, model-free et episodique. Contrairement a Q-Learning (Temporal Difference), il n'effectue aucune mise a jour pendant l'episode. Il attend la fin complete de l'episode pour mettre a jour les estimations de valeur."),

        para([
          bold("Les 3 phases d'un episode Monte Carlo :"),
        ]),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Generation : "), normal("jouer un episode complet en suivant la politique epsilon-greedy. Enregistrer chaque transition (etat s_t, action a_t, reward r_t).")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Retour cumule : "), normal("remonter l'episode de la fin vers le debut. Pour chaque pas t, calculer le retour cumule G_t = r_t + gamma * G_{t+1}. Ce retour integre toutes les consequences futures d'une action.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [bold("Mise a jour First-Visit : "), normal("pour chaque paire (etat, action) visitee pour la PREMIERE fois dans l'episode, mettre a jour la Q-table par moyenne incrementale : Q(s,a) += (G - Q(s,a)) / N(s,a), ou N(s,a) est le nombre de visites de (s,a).")]
        }),

        heading2("2.2 Comparaison avec Q-Learning"),
        makeTable(
          ["Aspect", "Monte Carlo", "Q-Learning (TD)"],
          [
            ["Quand met-il a jour ?", "Fin d'episode uniquement", "A chaque pas (en ligne)"],
            ["Learning rate (alpha) ?", "Non — moyenne exacte", "Oui — controle la vitesse"],
            ["Bootstrap ?", "Non — retour reel complet", "Oui — estime via max Q(s')"],
            ["Biais / Variance", "Faible biais, haute variance", "Haut biais, faible variance"],
            ["Adapte aux episodes longs ?", "Penalise (delai signal)", "Robuste"],
          ],
          [3000, 3500, 3500]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("2.3 Implications pour Taxi-v3"),
        para([
          normal("L'absence d'alpha est un enjeu majeur : les premieres experiences, quand l'agent explore au hasard, generent des retours tres negatifs (reward -200, -400). Ces valeurs sont integrees dans la moyenne et "),
          bold("ne peuvent pas etre corrigees"),
          normal(" par un parametre de learning rate. Les mauvaises estimations initiales pesent durablement sur la Q-table, surtout pour des (s,a) peu visites.")
        ]),
        para([
          normal("De plus, Taxi-v3 requiert une sequence de "),
          bold("2 actions cles separees"),
          normal(" (Pickup, puis Dropoff apres deplacement). Le signal de succes (+20) n'arrive qu'en fin d'episode, apres 13 a 200 pas. Monte Carlo doit attribuer ce credit a des decisions prises bien en amont — un credit assignment difficile.")
        ]),

        // ==================== 3. OPTIMISATION ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("3. Optimisation des hyperparametres"),

        para("Monte Carlo ne dispose que de deux leviers d'optimisation propres : gamma (discount factor) et epsilon_decay (vitesse de convergence vers l'exploitation). Nous appliquons la methode du grid search : faire varier un parametre a la fois en fixant tous les autres, mesurer l'impact sur les performances, puis retenir la meilleure valeur avant de passer au suivant."),

        para([
          bold("Protocole : "),
          normal("5 000 episodes d'entrainement, 100 episodes de test en mode greedy, seed aleatoire non fixe (resultats representatifs d'une execution reelle).")
        ]),

        // ---- 3.1 Grid search gamma ----
        heading2("3.1 Impact de gamma — Grid search"),

        para([
          normal("Gamma controle l'horizon temporel de l'agent : un gamma proche de 0 rend l'agent myope (seul le reward immediat compte) ; un gamma proche de 1 lui fait valoriser les rewards lointains. Nous testons 5 valeurs : [0.50, 0.70, 0.85, 0.95, 0.99], avec epsilon_decay=0.995 fixe le temps de cette experience.")
        ]),

        makeTable(
          ["Gamma", "Mean Steps", "Mean Reward", "Taux de succes", "Convergence"],
          [
            ["0.50", "200.0", "-200.0", "0.0 %", "N/A"],
            ["0.70", "200.0", "-200.0", "0.0 %", "N/A"],
            ["0.85", "200.0", "-200.0", "0.0 %", "N/A"],
            ["0.95", "194.3", "-301.7", "3.0 %", "N/A"],
            ["0.99", "198.1", "-575.9", "1.0 %", "N/A"],
          ],
          [1800, 1800, 1800, 1800, 1800],
          3
        ),
        new Paragraph({ spacing: { after: 200 } }),

        img("mc_grid_search_gamma.png", 6.5, 2.8),
        caption("Figure 1 : Impact de gamma sur les performances Monte Carlo (epsilon_decay=0.995 fixe)"),

        para([
          bold("Analyse des resultats :"),
        ]),
        para([
          normal("Le resultat dominant est que "),
          bold("gamma=0.95 est la seule valeur produisant un apprentissage significatif"),
          normal(" (3 % de succes). gamma=0.99 montre un signal faible (1 %) mais reste quasi-inutilisable. Toutes les valeurs inferieures donnent 0 % de succes.")
        ]),
        para([
          bold("Pourquoi gamma < 0.95 echoue : "),
          normal("avec un faible discount, le retour cumule G_t pour une action prise 10 pas avant le succes vaut : G = +20 * gamma^10. Avec gamma=0.85, cela donne 20 * 0.85^10 = 3.9 — un signal quasi-nul. L'agent ne peut pas associer ses decisions de deplacement a la recompense finale.")
        ]),
        para([
          bold("Pourquoi gamma=0.99 reste mauvais (-575.9 de reward) : "),
          normal("un gamma tres eleve amplifie les retours negatifs des longues sequences ratees. Un episode de 200 pas avec rewards -1 a chaque pas donne G = sum(-1 * 0.99^t) pour t de 0 a 199, soit environ -86 pour le premier pas. Ces valeurs tres negatives polluent la Q-table via la moyenne incrementale, sans possibilite de correction (pas d'alpha). Avec 50 000 episodes, l'agent accumule assez de retours positifs pour atteindre 1 % de succes, mais reste largement sous-optimal.")
        ]),
        para([
          bold("Valeur retenue : gamma = 0.95"),
          normal(" — unique valeur permettant un apprentissage. Elle offre un horizon temporel suffisant pour valoriser la recompense finale, sans amplifier excessivement les penalites des longs episodes ratees.")
        ]),

        // ---- 3.2 Grid search decay ----
        heading2("3.2 Impact d'epsilon_decay — Grid search"),

        para([
          normal("Avec gamma=0.95 fixe, nous faisons varier epsilon_decay sur [0.990, 0.995, 0.997, 0.999]. Ce parametre controle la vitesse de transition de l'exploration (epsilon=1.0) vers l'exploitation (epsilon=0.01). Un decay rapide fait passer l'agent en mode 'greedy' plus tot.")
        ]),

        makeTable(
          ["Epsilon Decay", "Episodes pour epsilon=0.01", "Mean Steps", "Mean Reward", "Taux de succes", "Convergence"],
          [
            ["0.990", "~460 ep.", "181.2", "-287.1", "10.0 %", "N/A"],
            ["0.995", "~920 ep.", "152.8", "-237.5", "25.0 %", "ep. 7 565"],
            ["0.997", "~1 530 ep.", "200.0", "-290.0", "0.0 %", "N/A"],
            ["0.999", "~4 605 ep.", "175.2", "-208.5", "13.0 %", "ep. 3 537"],
          ],
          [1600, 1600, 1600, 1600, 1600, 1400],
          1
        ),
        new Paragraph({ spacing: { after: 200 } }),

        img("mc_grid_search_decay.png", 6.5, 2.8),
        caption("Figure 2 : Impact d'epsilon_decay sur les performances Monte Carlo (gamma=0.95)"),

        para([
          bold("Analyse : "),
          normal("C'est le parametre le plus critique pour Monte Carlo. Avec 50 000 episodes, le gagnant est decay=0.995 avec 25 % de succes et convergence a l'episode 7 565 — un resultat surprenant par rapport a ce qu'on observe avec un budget de 5 000 episodes.")
        ]),
        para([
          normal("L'explication tient au rapport exploration/exploitation sur le budget total. Avec decay=0.995, l'agent passe en mode greedy vers l'episode 920. Il dispose alors d'environ 49 000 episodes pour exploiter et affiner sa politique — soit ~53x plus de temps d'exploitation qu'avec decay=0.999 (qui n'atteint son minimum qu'a l'episode 46 000, laissant seulement ~4 000 episodes d'exploitation sur 50 000).")
        ]),
        para([
          normal("Decay=0.997 donne paradoxalement 0 % de succes : il passe en exploitation a l'episode ~1 530, mais sa Q-table n'est pas encore assez bien construite a ce stade. Il se retrouve bloque dans une politique sous-optimale, avec trop peu d'exploration pour s'en sortir et trop peu d'episodes d'exploitation pour se rattraper.")
        ]),
        para([
          bold("Valeur retenue : epsilon_decay = 0.995"),
          normal(" — meilleur compromis entre construction de la Q-table et temps d'exploitation sur un budget de 50 000 episodes. Ce resultat illustre que l'hyperparametre optimal de Monte Carlo depend du budget d'entrainement disponible.")
        ]),

        // ==================== 4. RESULTATS ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("4. Resultats finaux — Parametres optimaux"),

        heading2("4.1 Configuration optimale"),
        para("En combinant les resultats des deux grid searches, les parametres optimaux pour Monte Carlo sur Taxi-v3 sont :"),

        makeTable(
          ["Parametre", "Valeur", "Source"],
          [
            ["Gamma (discount factor)", "0.95", "Grid search gamma — seule valeur non nulle"],
            ["Epsilon initial", "1.0", "Exploration totale au depart"],
            ["Epsilon minimum", "0.01", "Exploration residuelle en fin d'entrainement"],
            ["Epsilon decay", "0.995", "Grid search decay — meilleur ratio exploration/exploitation sur 50k ep."],
            ["Episodes d'entrainement", "50 000", "Budget suffisant pour la convergence MC"],
            ["Episodes de test", "100", "Mode greedy (epsilon=0)"],
          ],
          [2800, 1800, 5400]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("4.2 Courbes d'entrainement"),

        img("mc_training_optimized.png", 6.5, 5),
        caption("Figure 3 : Courbes d'entrainement Monte Carlo (gamma=0.95, epsilon_decay=0.995, 50 000 ep.)"),

        para([
          bold("Lecture des courbes : "),
          normal("La courbe de reward (lissee sur 100 episodes) progresse sur les 50 000 episodes, avec une transition visible vers l'episode 920 quand epsilon atteint son minimum et l'agent bascule en exploitation. Le bruit reste eleve (ecart-type = 367.6) du fait des episodes bimodaux : succes nets vs echecs complets. C'est la variance inherente a Monte Carlo sans alpha.")
        ]),

        heading2("4.3 Evaluation sur 8 metriques"),
        para([
          normal("Evaluation sur "),
          bold("100 episodes de test en mode greedy"),
          normal(" (epsilon=0, exploitation pure), apres 50 000 episodes d'entrainement.")
        ]),

        makeTable(
          ["#", "Metrique", "Definition", "Monte Carlo (opt.)"],
          [
            ["1", "Reward moyen", "mean(rewards)", "-238.5"],
            ["2", "Ecart-type reward", "std(rewards)", "367.6"],
            ["3", "Steps moyens", "mean(steps)", "169.9"],
            ["4", "Ecart-type steps", "std(steps)", "69.0"],
            ["5", "Taux de succes", "mean(reward > 0) * 100", "16.0 %"],
            ["6", "Reward / step", "sum(R) / sum(S)", "-1.404"],
            ["7", "Actes illegaux", "count(reward == -10)", "800"],
            ["8", "Episode de convergence", "1er ep. succes >= 25%/50ep", "ep. 5 761"],
          ],
          [400, 2200, 2800, 2600]
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("4.4 Interpretation des metriques"),

        para([
          bold("Metrique 5 — Taux de succes 16 % : "),
          normal("Monte Carlo resout le jeu dans 16 episodes sur 100. Ce chiffre, malgre sa modestie, represente un apprentissage reel : en brute-force, ce taux est de 0 %. L'agent a bien appris une politique partielle.")
        ]),
        para([
          bold("Metrique 2 — Ecart-type reward = 367.6 : "),
          normal("la variance tres elevee (vs 2.2 pour Q-Learning) revele la bimodalite de la politique MC : soit l'episode est un succes (reward positif, ~13 steps), soit c'est un echec complet (200 steps, reward -200). L'ecart-type depasse en valeur absolue le reward moyen, signe d'une politique instable qui n'a pas converge uniformement.")
        ]),
        para([
          bold("Metrique 7 — 800 actes illegaux : "),
          normal("Monte Carlo n'a pas appris a eviter systematiquement les Pickup/Dropoff illegaux. Avec TD learning, chaque penalite -10 est integree immediatement dans Q(s, Pickup). Avec Monte Carlo, cette penalite est diluee dans un retour cumule de fin d'episode et son signal est affaibli par les autres transitions de l'episode.")
        ]),
        para([
          bold("Metrique 8 — Convergence ep. 5 761 : "),
          normal("Monte Carlo necessite 10.9 fois plus d'episodes que Q-Learning (529) pour atteindre le seuil de 25 % de succes sur 50 episodes. Chaque episode de 200 pas ne produit qu'une mise a jour par (s,a). Q-Learning effectue une mise a jour par pas, soit jusqu'a 200x plus de mises a jour par episode long.")
        ]),

        // ==================== 5. COMPARAISON ====================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("5. Comparaison inter-algorithmes"),

        heading2("5.1 Tableau de synthese — 8 metriques"),
        para([
          normal("Comparaison sur 100 episodes de test en mode greedy. Chaque algorithme a ete entraine avec ses parametres optimaux. Monte Carlo : gamma=0.95, epsilon_decay=0.995, 50 000 episodes. Q-Learning : alpha=0.1, gamma=0.99, epsilon_decay=0.995. Brute-Force : aucun apprentissage.")
        ]),

        makeTable(
          ["Metrique", "Brute-Force", "Q-Learning", "Monte Carlo"],
          [
            ["Reward moyen",       "-774.3",  "8.4",     "-238.5"],
            ["Ecart-type reward",  "94.1",    "2.2",     "367.6"],
            ["Steps moyens",       "197.2",   "12.6",    "169.9"],
            ["Ecart-type steps",   "13.2",    "2.2",     "69.0"],
            ["Taux de succes",     "0.0 %",   "100.0 %", "16.0 %"],
            ["Reward / step",      "-3.927",  "0.671",   "-1.404"],
            ["Actes illegaux",     "6 424",   "0",       "800"],
            ["Convergence",        "N/A",     "ep. 529", "ep. 5 761"],
          ],
          [3000, 2000, 2000, 2000],
          1
        ),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("5.2 Courbes d'apprentissage comparatives"),

        img("mc_vs_ql_curves.png", 6.5, 3),
        caption("Figure 4 : Courbes d'apprentissage Q-Learning vs Monte Carlo (reward et steps lisses sur 100 ep.)"),

        img("mc_metrics_comparison.png", 6.5, 3),
        caption("Figure 5 : Comparaison des 4 metriques principales — Brute-Force, Q-Learning, Monte Carlo"),

        heading2("5.3 Analyse des ecarts"),

        para("La comparaison revele trois niveaux de performance distincts :"),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Brute-Force (reference inferieure) : "), normal("0 % de succes, 6 421 actes illegaux, reward/step de -3.921. Valeur de baseline uniquement, confirme que tout algorithme d'apprentissage apporte une amelioration mesurable.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 100 },
          children: [bold("Monte Carlo (apprentissage partiel) : "), normal("16 % de succes, ameliorations quantifiables : -238.5 de reward vs -774.3 pour BF (facteur 3.2), 169.9 steps vs 197.2 (reduction de 14 %), 800 actes illegaux vs 6 424 (reduction de 88 %). L'algorithme apprend une politique partielle mais ne converge pas completement.")]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [bold("Q-Learning (performance optimale) : "), normal("100 % de succes, 0 acte illegal, ecart-type de 2.4 (stabilite parfaite). La mise a jour en ligne (TD) avec learning rate alpha est decisvement superieure a la mise a jour episodique sans alpha pour un environnement avec credit assignment retarde comme Taxi-v3.")]
        }),

        heading2("5.4 Pourquoi Monte Carlo echoue a converger completement"),

        para("Les resultats permettent d'identifier 4 causes structurelles :"),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 120 },
          children: [
            bold("Pollution de la moyenne sans correction (ecart-type = 367.6 vs 2.2 pour QL) : "),
            normal("les premiers episodes exploratoires (reward -200 a -400) remplissent la Q-table de valeurs tres negatives. Sans alpha, Monte Carlo ne peut pas 'oublier' ces mauvaises estimations. Q-Learning avec alpha=0.1 les efface en ~10 mises a jour.")
          ]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 120 },
          children: [
            bold("Apprentissage lent des penalites -10 (800 actes illegaux vs 0) : "),
            normal("en TD learning, la penalite Pickup illegal est integree immediatement dans Q(s, Pickup). En MC, elle est diluee dans un retour cumule de 200 pas et son signal est attenue par les autres rewards de l'episode.")
          ]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 120 },
          children: [
            bold("Efficacite d'apprentissage superieure de Q-Learning (convergence ep. 5 761 vs 529) : "),
            normal("un episode de 200 pas produit ~200 transitions mais seulement 1 retour G par (s,a) premiere visite pour Monte Carlo. Q-Learning genere 200 mises a jour independantes par episode. MC necessite 10.9x plus d'episodes pour converger, meme avec un budget de 50 000 episodes.")
          ]
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 200 },
          children: [
            bold("Sensibilite extreme aux hyperparametres et au budget : "),
            normal("decay=0.997 donne 0 % de succes alors que 0.995 donne 25 %. Pire, le meilleur decay change selon le budget (0.999 optimal a 5k episodes, 0.995 a 50k). Cette fragilite contraste avec Q-Learning, robuste sur une large plage de parametres independamment du budget.")
          ]
        }),

        // ==================== 6. CONCLUSION ====================
        heading1("6. Conclusion"),

        para("Cette analyse approfondie de Monte Carlo First-Visit sur Taxi-v3 montre que l'algorithme peut apprendre une politique partielle mais ne converge pas completement sur cet environnement. Les resultats quantitatifs sur 8 metriques permettent de dresser le bilan suivant :"),

        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 100 },
          children: [bold("Ce qui fonctionne : "), normal("MC reduit les actes illegaux de 88 % (6 424 -> 800) et le reward moyen de 69 % par rapport au brute-force, prouvant un apprentissage reel.")]
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 100 },
          children: [bold("Ce qui bloque : "), normal("l'absence d'alpha, la mise a jour episodique et la sensibilite a epsilon_decay empechent une convergence complete sur un probleme a credit assignment retarde.")]
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 200 },
          children: [bold("Le parametre cle : "), normal("epsilon_decay=0.995 est optimal sur 50 000 episodes. Il equilibre exploration suffisante (~920 episodes) et exploitation prolongee (~49 000 episodes), permettant a la Q-table de se constituer puis de se raffiner. Ce parametre optimal depend du budget — avec 5k episodes, 0.999 serait preferable.")]
        }),

        para([
          normal("Monte Carlo est intrinsequement mieux adapte aux environnements ou les episodes sont courts, les rewards peu retardes, et ou l'on dispose d'un budget d'episodes important. Sur Taxi-v3, la longueur variable des episodes (13 a 200 pas) et la nature sequentielle de la tache favorisent des algorithmes a mise a jour en ligne comme Q-Learning.")
        ]),

        new Paragraph({ spacing: { before: 400 } }),
        para([
          bold("Parametres optimaux Monte Carlo retenus : "),
          normal("gamma=0.95, epsilon_decay=0.995, 50 000 episodes. Taux de succes final : 16 % (vs 0 % brute-force, 100 % Q-Learning). Convergence partielle a l'episode 5 761.")
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
