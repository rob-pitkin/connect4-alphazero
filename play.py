import os
import sys
import numpy as np
import pygame
import pyspiel
import torch

from mcts import MCTS
from model import ConnectFourNet

# Colors
BG_COLOR = (18, 20, 28)
PANEL_BG = (28, 32, 44)
BOARD_BLUE = (29, 78, 216)
BOARD_BORDER = (30, 58, 138)
EMPTY_SLOT = (18, 20, 28)

RED_DISC = (239, 68, 68)
RED_HIGHLIGHT = (252, 129, 129)
YELLOW_DISC = (250, 204, 21)
YELLOW_HIGHLIGHT = (254, 240, 138)

PRIOR_BAR_COLOR = (96, 165, 250)     # Soft Blue: Raw Network Prior P(s, a)
VISIT_BAR_COLOR = (52, 211, 153)     # Emerald Green: MCTS Visit Share pi(a | s)
TEXT_MAIN = (241, 245, 249)
TEXT_MUTED = (148, 163, 184)
ACCENT_GOLD = (251, 191, 36)


def board_grid_from_state(state) -> np.ndarray:
    """
    Returns a (6, 7) int array from OpenSpiel state where:
      0 = empty, 1 = Player 0 (Red), 2 = Player 1 (Yellow)
    Row 0 is the TOP row of the visual board, Row 5 is the BOTTOM row.
    """
    # Pass explicit player 0 so OpenSpiel doesn't default to current_player() (-4 when terminal)
    obs = np.array(state.observation_tensor(0)).reshape(3, 6, 7)
    grid = np.zeros((6, 7), dtype=int)
    # In OpenSpiel Connect Four observation_tensor, row 0 is the bottom row.
    # Flip vertically so row 0 is the top row for screen rendering.
    p0 = np.flip(obs[0], axis=0)
    p1 = np.flip(obs[1], axis=0)
    grid[p0 == 1.0] = 1
    grid[p1 == 1.0] = 2
    return grid


class ConnectFourVisualizer:
    def __init__(
        self,
        checkpoint_path: str = "connect4_best.pt",
        num_simulations: int = 160,
        sims_per_frame: int = 4,
    ):
        pygame.init()
        pygame.display.set_caption("Connect Four — AlphaZero Live MCTS Visualizer")
        self.width = 920
        self.height = 740
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("SF Pro Display, Helvetica, Arial", 24, bold=True)
        self.font_body = pygame.font.SysFont("SF Pro Text, Helvetica, Arial", 16)
        self.font_small = pygame.font.SysFont("SF Mono, Menlo, monospace", 13)
        self.font_bold_small = pygame.font.SysFont("SF Mono, Menlo, monospace", 13, bold=True)

        self.game = pyspiel.load_game("connect_four")
        self.checkpoint_path = checkpoint_path
        self.num_simulations = num_simulations
        self.sims_per_frame = sims_per_frame

        self.model = ConnectFourNet()
        self.load_weights()
        self.mcts = MCTS(self.model, c_puct=1.5, device="cpu")

        # Layout metrics
        self.board_left = 40
        self.board_top = 265
        self.cell_size = 76
        self.board_width = 7 * self.cell_size
        self.board_height = 6 * self.cell_size

        self.human_player = 0
        self.reset_game()

    def load_weights(self) -> None:
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
        else:
            self.weights_loaded = False
        self.model.eval()

    def reset_game(self) -> None:
        self.state = self.game.new_initial_state()
        self.last_action = None
        self.mcts_gen = None
        self.current_sim = 0
        self.raw_priors = np.zeros(7, dtype=np.float32)
        self.visit_probs = np.zeros(7, dtype=np.float32)
        self.action_q = np.zeros(7, dtype=np.float32)
        self.raw_value = 0.0
        self.search_q = 0.0

    def start_ai_search(self) -> None:
        self.mcts_gen = self.mcts.search_step_by_step(
            self.state, num_simulations=self.num_simulations, add_noise=False
        )
        self.current_sim = 0

    def step_ai_search(self) -> None:
        if self.mcts_gen is None:
            self.start_ai_search()

        root_node = None
        for _ in range(self.sims_per_frame):
            try:
                sim_idx, root_node, raw_priors, raw_val = next(self.mcts_gen)
                self.current_sim = sim_idx
                self.raw_priors = raw_priors
                self.raw_value = raw_val
            except StopIteration:
                # Search finished: play the most visited column
                best_action = int(np.argmax(self.visit_probs))
                self.state.apply_action(best_action)
                self.last_action = best_action
                self.mcts_gen = None
                return

        if root_node is not None:
            visits = np.zeros(7, dtype=np.float32)
            q_vals = np.zeros(7, dtype=np.float32)
            for action, child in root_node.children.items():
                visits[action] = child.visit_count
                q_vals[action] = -child.q_value  # From AI's perspective
            total = np.sum(visits)
            self.visit_probs = visits / total if total > 0 else visits
            self.action_q = q_vals
            self.search_q = root_node.q_value

    def get_hovered_col(self, mouse_x: int, mouse_y: int) -> int:
        if self.board_left <= mouse_x < self.board_left + self.board_width:
            if mouse_y >= self.board_top - 45:
                return (mouse_x - self.board_left) // self.cell_size
        return -1

    def draw(self, hovered_col: int) -> None:
        self.screen.fill(BG_COLOR)

        # 1. Header Banner
        curr_player = self.state.current_player()
        if self.state.is_terminal():
            returns = self.state.returns()
            if returns[self.human_player] > 0:
                status_text = "You Win! Press [R] to Play Again"
                status_color =VISIT_BAR_COLOR
            elif returns[self.human_player] < 0:
                status_text = "AlphaZero Wins! Press [R] to Play Again"
                status_color = RED_DISC
            else:
                status_text = "Draw Game! Press [R] to Play Again"
                status_color = ACCENT_GOLD
        elif curr_player == self.human_player:
            disc_name = "Red" if self.human_player == 0 else "Yellow"
            status_text = f"Your Turn ({disc_name}) — Click a column or press 0-6"
            status_color = TEXT_MAIN
        else:
            status_text = (
                f"AlphaZero Deliberating... MCTS Simulation {self.current_sim}/{self.num_simulations}"
            )
            status_color = ACCENT_GOLD

        title_surf = self.font_title.render(status_text, True, status_color)
        self.screen.blit(title_surf, (self.board_left, 18))

        sub_text = (
            f"[R] Reset   [Tab] Swap Side   [Up/Down] MCTS Budget: {self.num_simulations} sims   "
            f"Model: {'connect4_best.pt' if self.weights_loaded else 'Untrained (Random Weights)'}"
        )
        sub_surf = self.font_small.render(sub_text, True, TEXT_MUTED)
        self.screen.blit(sub_surf, (self.board_left, 50))

        # 2. Live MCTS Bar Chart Panel (Above the 7 columns)
        chart_top = 80
        chart_bottom = 225
        max_bar_h = 105
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (self.board_left, chart_top, self.board_width, chart_bottom - chart_top + 25),
            border_radius=10,
        )

        legal_actions = set(self.state.legal_actions()) if not self.state.is_terminal() else set()

        for c in range(7):
            col_center_x = self.board_left + c * self.cell_size + self.cell_size // 2
            prior_p = float(self.raw_priors[c])
            visit_p = float(self.visit_probs[c])

            # Left bar: Prior P(s, a)
            prior_h = int(prior_p * max_bar_h)
            prior_rect = pygame.Rect(
                col_center_x - 20, chart_bottom - prior_h, 17, max(2, prior_h)
            )
            pygame.draw.rect(self.screen, PRIOR_BAR_COLOR, prior_rect, border_radius=3)

            # Right bar: MCTS Visit Share pi(a | s)
            visit_h = int(visit_p * max_bar_h)
            visit_rect = pygame.Rect(
                col_center_x + 3, chart_bottom - visit_h, 17, max(2, visit_h)
            )
            pygame.draw.rect(self.screen, VISIT_BAR_COLOR, visit_rect, border_radius=3)

            # Percentage & Q-value labels
            pct_str = f"{visit_p * 100:3.0f}%"
            pct_surf = self.font_bold_small.render(pct_str, True, TEXT_MAIN)
            self.screen.blit(
                pct_surf, (col_center_x - pct_surf.get_width() // 2, chart_bottom - max(prior_h, visit_h) - 18)
            )

            q_str = f"Q:{self.action_q[c]:+.2f}" if visit_p > 0 else f"Col {c}"
            q_surf = self.font_small.render(q_str, True, TEXT_MUTED)
            self.screen.blit(
                q_surf, (col_center_x - q_surf.get_width() // 2, chart_bottom + 5)
            )

        # Hover preview disc for Human turn
        if (
            not self.state.is_terminal()
            and curr_player == self.human_player
            and hovered_col in legal_actions
        ):
            cx = self.board_left + hovered_col * self.cell_size + self.cell_size // 2
            cy = self.board_top - 10
            color = RED_DISC if self.human_player == 0 else YELLOW_DISC
            pygame.draw.circle(self.screen, color, (cx, cy), 14)

        # 3. Connect Four Board Grid
        board_rect = pygame.Rect(
            self.board_left, self.board_top, self.board_width, self.board_height
        )
        pygame.draw.rect(self.screen, BOARD_BLUE, board_rect, border_radius=14)
        pygame.draw.rect(self.screen, BOARD_BORDER, board_rect, width=3, border_radius=14)

        grid = board_grid_from_state(self.state)
        for r in range(6):
            for c in range(7):
                cx = self.board_left + c * self.cell_size + self.cell_size // 2
                cy = self.board_top + r * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 2 - 8

                cell_val = grid[r, c]
                if cell_val == 0:
                    pygame.draw.circle(self.screen, EMPTY_SLOT, (cx, cy), radius)
                elif cell_val == 1:
                    pygame.draw.circle(self.screen, RED_DISC, (cx, cy), radius)
                    pygame.draw.circle(self.screen, RED_HIGHLIGHT, (cx - 6, cy - 6), radius // 3)
                else:
                    pygame.draw.circle(self.screen, YELLOW_DISC, (cx, cy), radius)
                    pygame.draw.circle(self.screen, YELLOW_HIGHLIGHT, (cx - 6, cy - 6), radius // 3)

        # Highlight column of last move
        if self.last_action is not None:
            for r in range(6):
                if grid[r, self.last_action] != 0:
                    cx = self.board_left + self.last_action * self.cell_size + self.cell_size // 2
                    cy = self.board_top + r * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(self.screen, ACCENT_GOLD, (cx, cy), self.cell_size // 2 - 6, width=3)
                    break

        # 4. Right-Side Panel: Win Evaluation Gauge & Legend
        side_left = self.board_left + self.board_width + 30
        side_w = self.width - side_left - 30
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (side_left, chart_top, side_w, self.board_top + self.board_height - chart_top),
            border_radius=12,
        )

        # Legend
        leg_title = self.font_body.render("MCTS Live Telemetry", True, TEXT_MAIN)
        self.screen.blit(leg_title, (side_left + 20, chart_top + 18))

        pygame.draw.rect(self.screen, PRIOR_BAR_COLOR, (side_left + 20, chart_top + 52, 16, 16), border_radius=3)
        self.screen.blit(
            self.font_small.render("Prior P(s,a) (Intuition)", True, TEXT_MUTED),
            (side_left + 44, chart_top + 52),
        )

        pygame.draw.rect(self.screen, VISIT_BAR_COLOR, (side_left + 20, chart_top + 78, 16, 16), border_radius=3)
        self.screen.blit(
            self.font_small.render("Visits π(a|s) (Search)", True, TEXT_MUTED),
            (side_left + 44, chart_top + 78),
        )

        # Value Gauge (-1.0 to +1.0 from AI's perspective)
        gauge_top = chart_top + 135
        gauge_h = 340
        gauge_x = side_left + 45
        gauge_w = 36

        pygame.draw.rect(
            self.screen, BG_COLOR, (gauge_x, gauge_top, gauge_w, gauge_h), border_radius=6
        )
        mid_y = gauge_top + gauge_h // 2

        # Fill bar from center based on search_q
        clamped_q = max(-1.0, min(1.0, float(self.search_q)))
        fill_h = int(abs(clamped_q) * (gauge_h // 2))
        if clamped_q >= 0:
            pygame.draw.rect(
                self.screen,
                VISIT_BAR_COLOR,
                (gauge_x, mid_y - fill_h, gauge_w, fill_h),
                border_radius=4,
            )
        else:
            pygame.draw.rect(
                self.screen,
                RED_DISC,
                (gauge_x, mid_y, gauge_w, fill_h),
                border_radius=4,
            )

        # Center 0.0 line
        pygame.draw.line(
            self.screen, TEXT_MAIN, (gauge_x - 6, mid_y), (gauge_x + gauge_w + 6, mid_y), 2
        )

        # Labels next to gauge
        self.screen.blit(
            self.font_small.render("+1.0 (AI Win)", True, VISIT_BAR_COLOR),
            (gauge_x + gauge_w + 14, gauge_top),
        )
        self.screen.blit(
            self.font_small.render(" 0.0 (Even)", True, TEXT_MUTED),
            (gauge_x + gauge_w + 14, mid_y - 8),
        )
        self.screen.blit(
            self.font_small.render("-1.0 (You Win)", True, RED_DISC),
            (gauge_x + gauge_w + 14, gauge_top + gauge_h - 16),
        )

        # Numeric readout at bottom of side panel
        v_str = f"Net V(s):   {self.raw_value:+.3f}"
        q_str = f"MCTS Q(s):  {self.search_q:+.3f}"
        self.screen.blit(
            self.font_bold_small.render(v_str, True, PRIOR_BAR_COLOR),
            (side_left + 20, gauge_top + gauge_h + 30),
        )
        self.screen.blit(
            self.font_bold_small.render(q_str, True, VISIT_BAR_COLOR),
            (side_left + 20, gauge_top + gauge_h + 54),
        )

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            hovered_col = self.get_hovered_col(mouse_x, mouse_y)
            curr_player = self.state.current_player()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.load_weights()
                        self.reset_game()
                    elif event.key == pygame.K_TAB:
                        self.human_player = 1 - self.human_player
                        self.reset_game()
                    elif event.key == pygame.K_UP:
                        self.num_simulations = min(1000, self.num_simulations + 40)
                    elif event.key == pygame.K_DOWN:
                        self.num_simulations = max(20, self.num_simulations - 40)
                    elif (
                        not self.state.is_terminal()
                        and curr_player == self.human_player
                        and pygame.K_0 <= event.key <= pygame.K_6
                    ):
                        col = event.key - pygame.K_0
                        if col in self.state.legal_actions():
                            self.state.apply_action(col)
                            self.last_action = col
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not self.state.is_terminal()
                    and curr_player == self.human_player
                ):
                    if hovered_col in self.state.legal_actions():
                        self.state.apply_action(hovered_col)
                        self.last_action = hovered_col

            # Step AI MCTS search a few simulations per frame for smooth 60 FPS bar animation
            if not self.state.is_terminal() and self.state.current_player() != self.human_player:
                self.step_ai_search()

            self.draw(hovered_col)
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "connect4_best.pt"
    viz = ConnectFourVisualizer(checkpoint_path=ckpt, num_simulations=160, sims_per_frame=4)
    viz.run()
