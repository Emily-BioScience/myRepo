import pygame
from pygame.sprite import Sprite

class Bullet(Sprite):

    def __init__(self, ai_settings, screen, ship):
        super(Bullet, self).__init__()
        self.screen = screen

        # create a bullet
        self.rect = pygame.Rect(0, 0, ai_settings.bullet_width, ai_settings.bullet_height)
        self.rect.centerx = ship.rect.centerx  # update, bullet from the ship
        self.rect.top = ship.rect.top  # update, bullet from the ship

        self.y = float(self.rect.y)
        self.color = ai_settings.bullet_color
        self.speed = ai_settings.bullet_speed


    def update(self):
        self.y -= self.speed
        self.rect.y = self.y


    def draw_bullet(self):
        pygame.draw.rect(self.screen, self.color, self.rect)