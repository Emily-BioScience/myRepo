import pygame
from pygame.sprite import Group

from settings import Settings
from ship import Ship
import game_functions as gf

def run_game():
    #    create
    settings
    ai_settings = Settings()

    # create a screen
    pygame.init()
    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption(ai_settings.caption)

    # create a ship
    ship = Ship(ai_settings, screen)

    # create the bullets
    bullets = Group()

    # start loop
    while True:
        # monitor and keyboard and mouse event
        gf.check_events(ai_settings, screen, ship, bullets)
        ship.update()
        gf.update_bullets(bullets)

        # update screen every time
        gf.update_screen(ai_settings, screen, ship, bullets)

run_game()