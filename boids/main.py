import pygame
from  boids.config import Config
from  boids.simulation import Simulation
from boids.renderer import Renderer

def main():
    cfg = Config()
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Swarm Playground — algorithm sandbox")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    sim = Simulation(cfg)
    renderer = Renderer(cfg)
    paused = False
    running = True

    while running:
        dt = min(clock.tick(cfg.fps_limit) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim.reset()
                    renderer.clear()
                elif event.key == pygame.K_c:
                    renderer.clear()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    sim.add_particles(100)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim.remove_particles(100)

        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed(3)
        mouse = pygame.mouse.get_pos()
        mouse_mode = 1 if mouse_buttons[0] else (-1 if mouse_buttons[2] else 0)

        if not paused:
            sim.step(
                dt,
                mouse_position=mouse,
                mouse_mode=mouse_mode,
            )

        renderer.draw(screen, sim.positions, sim.velocities)

        status = "PAUSED" if paused else "RUNNING"
        lines = [
            f"{status} | particles: {sim.count} | FPS: {clock.get_fps():.0f}",
            "SPACE pause  R reset  C clear trails  +/- add/remove  ESC quit",
            "Mouse: left/right buttons are passed to behavior.py as input",
        ]
        y = 12
        for line in lines:
            surf = font.render(line, True, (180, 195, 215))
            screen.blit(surf, (14, y))
            y += 21

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
