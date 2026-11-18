import pygame

from  visual.default import Configuration
from  visual.simulation import Simulation
from  visual.renderer import Renderer
from  behavior import SwarmBehavior

def run(behavior:SwarmBehavior, config:Configuration | None = None ):

    cfg = config if config else Configuration()
    
    pygame.init()
    pygame.display.set_caption("Swarm Playground — algorithm sandbox")
    
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    sim = Simulation(config=cfg)
    renderer = Renderer(config=cfg)
    
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

        #keys = pygame.key.get_pressed()
        
        mouse_buttons = pygame.mouse.get_pressed(3)
        mouse = pygame.mouse.get_pos()
        mouse_mode = 1 if mouse_buttons[0] else (-1 if mouse_buttons[2] else 0)

        if not paused:
            sim.step(
                dt=dt,
                update_velocities=behavior.update_velocities,
                mouse_position=mouse,
                mouse_mode=mouse_mode,
            )

        renderer.draw(
            screen,
            sim.positions,
            sim.velocities,
            sim.obstacles,
            targets=sim.targets,
            grid=behavior.get_grid(), #type:ignore
        )

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