import pygame
import numpy as np

# Constants
GRID_SIZE = 28
PIXEL_SIZE = 25
CANVAS_WIDTH = GRID_SIZE * PIXEL_SIZE
WINDOW_WIDTH = CANVAS_WIDTH + 300
WINDOW_HEIGHT = CANVAS_WIDTH
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)




class DrawingCanvas:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.lastPoint = None

    def draw_grid(self):
        for x in range(GRID_SIZE + 1):
            pygame.draw.line(
                self.screen, GRAY, (x * PIXEL_SIZE, 0), (x * PIXEL_SIZE, CANVAS_WIDTH)
            )
        for y in range(GRID_SIZE + 1):
            pygame.draw.line(
                self.screen, GRAY, (0, y * PIXEL_SIZE), (CANVAS_WIDTH, y * PIXEL_SIZE)
            )

    def draw_pixels(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                val = self.canvas[y, x]
                if val > 0:
                    rect = pygame.Rect(
                        x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE
                    )
                    pygame.draw.rect(self.screen, (val, val, val), rect)

    def draw(self):
        running = True
        while running:
            self.clock.tick(60)
            self.screen.fill(BLACK)

            mouse_pressed = pygame.mouse.get_pressed()[0]
            if mouse_pressed:
                x, y = pygame.mouse.get_pos()
                if x < CANVAS_WIDTH:
                    grid_x = x // PIXEL_SIZE
                    grid_y = y // PIXEL_SIZE
                    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                        last = self.lastPoint

                        # ! TODO
                        # if(last):
                        #     l = len(range(last[0], grid_x))
                        #     for i in range(last[0], grid_x):

                        #     pass
                            # color_pixel
                            
                        self.color_pixel(grid_x, grid_y)
                        self.lastPoint = (grid_x, grid_y)


            self.draw_pixels()
            self.draw_grid()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

    def color_pixel(self, grid_x, grid_y):
        self.canvas[grid_y, grid_x] = 255
        #
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                self.canvas[ny, nx] = max(self.canvas[ny, nx], 150)
        #
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                self.canvas[ny, nx] = max(self.canvas[ny, nx], 100)
        # #
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                self.canvas[ny, nx] = max(self.canvas[ny, nx], 30)

    def get_canvas(self):
        return self.canvas.copy()


