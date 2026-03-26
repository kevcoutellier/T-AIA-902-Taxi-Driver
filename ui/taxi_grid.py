import tkinter as tk
from tkinter import Canvas


class TaxiGridVisualization(Canvas):
    """
    Visualisation de la grille 5x5 du jeu Taxi.

    La grille affiche:
    - Le taxi (carré jaune)
    - Le passager (cercle bleu)
    - La destination (cercle vert)
    - Les emplacements spéciaux: R, G, Y, B (coins)
    - Les murs (lignes noires)
    """

    def __init__(self, parent, cell_size=80):
        """
        Args:
            parent: Widget parent
            cell_size: Taille d'une cellule en pixels
        """
        self.cell_size = cell_size
        self.grid_size = 5
        canvas_size = self.grid_size * cell_size

        super().__init__(
            parent,
            width=canvas_size,
            height=canvas_size,
            bg='white',
            highlightthickness=2,
            highlightbackground='#333'
        )

        # Couleurs
        self.colors = {
            'grid': '#CCCCCC',
            'taxi': '#FFD700',
            'passenger': '#4169E1',
            'destination': '#32CD32',
            'wall': '#333333',
            'location_R': '#FF6B6B',
            'location_G': '#4ECDC4',
            'location_Y': '#FFE66D',
            'location_B': '#95E1D3'
        }

        # Positions des emplacements spéciaux dans la grille Taxi
        # (row, col)
        self.special_locations = {
            'R': (0, 0),  # Rouge - coin haut-gauche
            'G': (0, 4),  # Vert - coin haut-droit
            'Y': (4, 0),  # Jaune - coin bas-gauche
            'B': (4, 3)   # Bleu - bas-droite (pas exactement coin)
        }

        # Murs horizontaux et verticaux du Taxi environment
        # Format: (row, col, direction) où direction = 'h' (horizontal) ou 'v' (vertical)
        self.walls = [
            # Murs verticaux (|)
            (0, 0, 'v'), (1, 0, 'v'),  # Gauche de R
            (0, 2, 'v'), (1, 2, 'v'),  # Milieu haut
            (3, 0, 'v'), (4, 0, 'v'),  # Gauche de Y
            (3, 2, 'v'), (4, 2, 'v'),  # Milieu bas
        ]

        self.draw_grid()

    def draw_grid(self):
        """Dessine la grille de base."""
        # Lignes de grille
        for i in range(self.grid_size + 1):
            # Lignes horizontales
            self.create_line(
                0, i * self.cell_size,
                self.grid_size * self.cell_size, i * self.cell_size,
                fill=self.colors['grid'], width=1
            )
            # Lignes verticales
            self.create_line(
                i * self.cell_size, 0,
                i * self.cell_size, self.grid_size * self.cell_size,
                fill=self.colors['grid'], width=1
            )

        # Emplacements spéciaux (R, G, Y, B)
        for label, (row, col) in self.special_locations.items():
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2

            # Fond coloré
            self.create_rectangle(
                col * self.cell_size + 5,
                row * self.cell_size + 5,
                (col + 1) * self.cell_size - 5,
                (row + 1) * self.cell_size - 5,
                fill=self.colors[f'location_{label}'],
                outline='',
                tags='location'
            )

            # Label
            self.create_text(
                x, y,
                text=label,
                font=('Arial', 20, 'bold'),
                fill='white',
                tags='location'
            )

        # Murs
        for row, col, direction in self.walls:
            if direction == 'v':  # Mur vertical (droite de la cellule)
                x = (col + 1) * self.cell_size
                y1 = row * self.cell_size
                y2 = (row + 1) * self.cell_size
                self.create_line(x, y1, x, y2, fill=self.colors['wall'], width=4, tags='wall')
            else:  # Mur horizontal (bas de la cellule)
                y = (row + 1) * self.cell_size
                x1 = col * self.cell_size
                x2 = (col + 1) * self.cell_size
                self.create_line(x1, y, x2, y, fill=self.colors['wall'], width=4, tags='wall')

    def decode_state(self, state):
        """
        Décode l'état numérique du Taxi en composants.

        L'état Taxi est encodé comme un entier qui combine:
        - Position du taxi (25 positions dans 5x5)
        - Position du passager (5 emplacements: R, G, Y, B, ou dans taxi)
        - Destination (4 emplacements: R, G, Y, B)

        Args:
            state: État entier (0-499)

        Returns:
            tuple: (taxi_row, taxi_col, passenger_loc, destination)
        """
        # Décodage de l'état Taxi-v3
        # state = taxi_row * 100 + taxi_col * 20 + passenger_loc * 4 + destination

        destination = state % 4
        state = state // 4

        passenger_loc = state % 5
        state = state // 5

        taxi_col = state % 5
        taxi_row = state // 5

        return taxi_row, taxi_col, passenger_loc, destination

    def get_location_position(self, location_index):
        """
        Retourne la position (row, col) d'un emplacement.

        Args:
            location_index: 0=R, 1=G, 2=Y, 3=B

        Returns:
            tuple: (row, col)
        """
        locations = ['R', 'G', 'Y', 'B']
        if location_index < 4:
            location_label = locations[location_index]
            return self.special_locations[location_label]
        return None

    def draw_state(self, state):
        """
        Dessine l'état actuel du jeu.

        Args:
            state: État du jeu (entier 0-499)
        """
        # Efface les éléments mobiles précédents
        self.delete('taxi', 'passenger', 'dest_marker')

        # Décode l'état
        taxi_row, taxi_col, passenger_loc, destination = self.decode_state(state)

        # Position du taxi en pixels
        taxi_x = taxi_col * self.cell_size + self.cell_size // 2
        taxi_y = taxi_row * self.cell_size + self.cell_size // 2

        # Dessine le taxi (carré jaune)
        taxi_size = self.cell_size // 3
        self.create_rectangle(
            taxi_x - taxi_size // 2,
            taxi_y - taxi_size // 2,
            taxi_x + taxi_size // 2,
            taxi_y + taxi_size // 2,
            fill=self.colors['taxi'],
            outline='black',
            width=2,
            tags='taxi'
        )

        # Label "TAXI"
        self.create_text(
            taxi_x,
            taxi_y,
            text='T',
            font=('Arial', 16, 'bold'),
            fill='black',
            tags='taxi'
        )

        # Dessine le passager (si pas dans le taxi)
        if passenger_loc < 4:
            passenger_pos = self.get_location_position(passenger_loc)
            if passenger_pos:
                p_row, p_col = passenger_pos
                p_x = p_col * self.cell_size + self.cell_size // 2
                p_y = p_row * self.cell_size + self.cell_size // 2

                # Cercle bleu pour le passager
                radius = self.cell_size // 6
                self.create_oval(
                    p_x - radius, p_y - radius,
                    p_x + radius, p_y + radius,
                    fill=self.colors['passenger'],
                    outline='white',
                    width=2,
                    tags='passenger'
                )
                self.create_text(
                    p_x, p_y,
                    text='P',
                    font=('Arial', 12, 'bold'),
                    fill='white',
                    tags='passenger'
                )
        else:
            # Passager dans le taxi - indicateur
            self.create_text(
                taxi_x,
                taxi_y - taxi_size // 2 - 10,
                text='👤',
                font=('Arial', 12),
                tags='passenger'
            )

        # Marque la destination (étoile verte)
        dest_pos = self.get_location_position(destination)
        if dest_pos:
            d_row, d_col = dest_pos
            d_x = d_col * self.cell_size + self.cell_size // 2
            d_y = d_row * self.cell_size + self.cell_size // 2

            # Étoile pour la destination
            self.create_text(
                d_x,
                d_y - self.cell_size // 3,
                text='★',
                font=('Arial', 24),
                fill=self.colors['destination'],
                tags='dest_marker'
            )

    def clear(self):
        """Efface tous les éléments mobiles."""
        self.delete('taxi', 'passenger', 'dest_marker')
