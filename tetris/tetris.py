import pygame, random, math

# ---------- Config ----------
WIDTH, HEIGHT = 300, 600
BLOCK_SIZE = 30
cols, rows = WIDTH // BLOCK_SIZE, HEIGHT // BLOCK_SIZE
LINES_PER_LEVEL = 10

BLACK=(0,0,0); GRAY=(70,70,70)
COLORS=[(0,255,255),(0,0,255),(255,165,0),(255,255,0),(0,255,0),(128,0,128),(255,0,0)]
SHAPES=[
    [[1,1,1,1]],                   # I
    [[1,1,1],[0,1,0]],             # T
    [[1,1,0],[0,1,1]],             # S
    [[0,1,1],[1,1,0]],             # Z
    [[1,1],[1,1]],                 # O
    [[1,1,1],[1,0,0]],             # L
    [[1,1,1],[0,0,1]],             # J
]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

grid = [[BLACK for _ in range(cols)] for _ in range(rows)]

class Piece:
    def __init__(self, shape):
        self.shape = [row[:] for row in shape]
        self.color = random.choice(COLORS)
        self.x = cols // 2 - len(shape[0]) // 2
        self.y = 0
    def rotate(self):
        self.shape = [list(r) for r in zip(*self.shape[::-1])]

def valid_position(p, dx=0, dy=0):
    for y,row in enumerate(p.shape):
        for x,val in enumerate(row):
            if not val: continue
            nx = p.x + x + dx
            ny = p.y + y + dy
            if nx < 0 or nx >= cols or ny >= rows: return False
            if ny >= 0 and grid[ny][nx] != BLACK: return False
    return True

def merge_piece(p):
    for y,row in enumerate(p.shape):
        for x,val in enumerate(row):
            if val:
                grid[p.y+y][p.x+x] = p.color

def clear_rows():
    """Clears full rows; returns number cleared."""
    global grid
    kept = [row for row in grid if any(c==BLACK for c in row)]
    cleared = rows - len(kept)
    while len(kept) < rows:
        kept.insert(0, [BLACK]*cols)
    grid = kept
    return cleared

def score_for_lines(n):
    return {0:0, 1:100, 2:300, 3:500, 4:800}.get(n, 0)

def fall_interval_ms(level):
    # Classic guideline: time *= 0.9^(level-1); base ~800ms
    return max(100, int(800 * (0.9 ** (level-1))))

# ---------- Game State ----------
running = True
current = Piece(random.choice(SHAPES))
fall_timer = 0
level = 1
total_lines = 0
score = 0

while running:
    dt = clock.tick(60)
    fall_timer += dt

    # input
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_LEFT and valid_position(current, dx=-1):
                current.x -= 1
            elif e.key == pygame.K_RIGHT and valid_position(current, dx=1):
                current.x += 1
            elif e.key == pygame.K_DOWN and valid_position(current, dy=1):
                current.y += 1
            elif e.key == pygame.K_UP:
                old = [row[:] for row in current.shape]
                current.rotate()
                if not valid_position(current):  # simple wall-kick rollback
                    current.shape = old

    # gravity
    if fall_timer >= fall_interval_ms(level):
        if valid_position(current, dy=1):
            current.y += 1
        else:
            merge_piece(current)
            cleared = clear_rows()
            # update scoring/level
            score += score_for_lines(cleared) * level
            total_lines += cleared
            level = max(1, total_lines // LINES_PER_LEVEL + 1)

            current = Piece(random.choice(SHAPES))
            if not valid_position(current):
                running = False  # game over
        fall_timer = 0

    # draw
    screen.fill(BLACK)
    # grid
    for y in range(rows):
        for x in range(cols):
            pygame.draw.rect(screen, grid[y][x], (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, GRAY, (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    # current piece
    for y,row in enumerate(current.shape):
        for x,val in enumerate(row):
            if val:
                pygame.draw.rect(
                    screen, current.color,
                    ((current.x+x)*BLOCK_SIZE, (current.y+y)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                )

    # HUD
    hud = font.render(f"Score: {score}   Lines: {total_lines}   Level: {level}", True, (220,220,220))
    screen.blit(hud, (8, 8))

    pygame.display.flip()

pygame.quit()
