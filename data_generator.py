import torch 

import math
import random
from config import DEVICE



class DataGenerator:
    def __init__(self,n,m,circles=(5,5),radius=(15,10),seed=None):
        """
        n: int
            number of rows
        m: int
            number of columns
        circles: tuple
            mean of the distribution, standard deviation of the distribution
        radius: tuple
            mean of the distribution, standard deviation of the distribution
        seed: int
            if set to None, the its value is random
        """
        self.n = n
        self.m = m
        self.circles = circles
        self.radius = radius


        self.seed = random.randint(0, 100000000) if seed is None else seed

        self.device = DEVICE

    def generate_data(self,batch = 1):
        """
        n_circles as first value flat  
        radius as normal
        """

        # n_circles = torch.round(torch.normal(self.circles[0], self.circles[1], (batch,), device=self.device))
        # max_n_circles = int(n_circles.max().item()) # maximum number of circles in the batch

        max_n_circles = self.circles[0]



        # TODO modifiy this to account for diffrent number of circles by changing mean and std over the n_circles as tensor value for them 
        r = torch.round(torch.normal(self.radius[0], self.radius[1], (batch,max_n_circles,), device=self.device))
        # idx = torch.arange(max_n_circles, device=self.device).view(1, max_n_circles).expand(batch, max_n_circles)  # (x,k)
        # valid = idx < n_circles.view(batch, 1)  

        # maybe remove just make sure whoe circles are in idk how it would influence later generation with neg nums 
        # range_x = (self.n - 2 * r).clamp(min=1).to(torch.float32)  # shape (k,)
        # range_y = (self.m - 2 * r).clamp(min=1).to(torch.float32)
        # cx = (r.to(torch.float32) + torch.floor(torch.rand((batch,max_n_circles), device=self.device) * range_x)).to(torch.int16)  # (k,)
        # cy = (r.to(torch.float32) + torch.floor(torch.rand((batch,max_n_circles), device=self.device) * range_y)).to(torch.int16)  # (k,)

        cx = torch.floor(torch.rand((batch,max_n_circles), device=self.device) * self.n).to(torch.int16)  # (k,)
        cy = torch.floor(torch.rand((batch,max_n_circles), device=self.device) * self.m).to(torch.int16)  # (k,)


        xs = torch.arange(self.n, device=self.device, dtype=torch.int16).view(self.n, 1, 1)   


        ys = torch.arange(self.m, device=self.device, dtype=torch.int16).view(1, self.m, 1)   
        cx_f = cx.view(batch,1, 1, max_n_circles).to(torch.float16)
        cy_f = cy.view(batch, 1, 1, max_n_circles).to(torch.float16)
        r2 = (r.view(batch, 1, 1, max_n_circles).to(torch.int16)) ** 2

        mask_all = (xs - cx_f) ** 2 + (ys - cy_f) ** 2 <= r2


        return mask_all.any(dim=3) 

    # def generate_data(self,batch = 1):
    #     n_circles = torch.round(torch.normal(self.circles[0], self.circles[1], (batch,), device=self.device))
    #     max_n_circles = int(n_circles.max().item()) # maximum number of circles in the batch


    #     if max_n_circles <= 0:
    #         return torch.zeros((batch, self.n, self.m), dtype=torch.bool)


    #     r = torch.round(torch.normal(self.radius[0], self.radius[1], (batch,max_n_circles,), device=self.device))
    #     idx = torch.arange(max_n_circles, device=self.device).view(1, max_n_circles).expand(batch, max_n_circles)  # (x,k)
    #     valid = idx < n_circles.view(batch, 1)  


    #     range_x = (self.n - 2 * r).clamp(min=1).to(torch.float32)  # shape (k,)
    #     range_y = (self.m - 2 * r).clamp(min=1).to(torch.float32)


    #     cx = (r.to(torch.float32) + torch.floor(torch.rand((batch,max_n_circles), device=self.device) * range_x)).to(torch.long)  # (k,)
    #     cy = (r.to(torch.float32) + torch.floor(torch.rand((batch,max_n_circles), device=self.device) * range_y)).to(torch.long)  # (k,)


    #     xs = torch.arange(self.n, device=self.device, dtype=torch.float32).view(self.n, 1, 1)   
    #     ys = torch.arange(self.m, device=self.device, dtype=torch.float32).view(1, self.m, 1)   
    #     cx_f = cx.view(batch,1, 1, max_n_circles).to(torch.float32)
    #     cy_f = cy.view(batch, 1, 1, max_n_circles).to(torch.float32)
    #     r2 = (r.view(batch, 1, 1, max_n_circles).to(torch.float32)) ** 2

    #     mask_all = (xs - cx_f) ** 2 + (ys - cy_f) ** 2 <= r2
    #     mask_all = mask_all & valid.view(batch,1,1, max_n_circles) 
    #     grid = mask_all.any(dim=3)  # (n,m) bool on device

    #     return grid
    

    def generate_data_cpu(self):
        #for c in range()
        
        """
        n_circles as normal 
        radius as normal
        """

        grid = torch.zeros((self.n, self.m), dtype=torch.bool)
        circles = []



        for _ in range(int(random.gauss(self.circles[0], self.circles[1]))):

            r = int(round(random.gauss(self.radius[0], self.radius[1])))
            r = max(1, min(r, min(self.n, self.m) // 2))



            # sample center uniformly but ensure full circle fits
            if self.n - 2 * r <= 0:
                cx = self.n // 2
            else:
                cx = int(torch.randint(r, self.n - r, (1,)).item())

            if self.m - 2 * r <= 0:
                cy = self.m // 2
            else:
                cy = int(torch.randint(r, self.m - r, (1,)).item())


            circles.append((cx, cy, r))

        # print('circles:', circles   )

        # rasterize circles into grid (overlap allowed)
        xs = torch.arange(self.n, dtype=torch.int32).view(self.n, 1)  # (n,1)
        ys = torch.arange(self.m, dtype=torch.int32).view(1, self.m)  # (1,m)
        for cx, cy, r in circles:
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
            grid[mask] = 1

        # return [[0] * self.n] * self.m  # Placeholder implementation
        return grid.cpu().tolist()
    

if __name__ == "__main__":
    dg = DataGenerator(10,10,(3,5),(1,2), seed=42)
    data = dg.generate_data(batch=50)

    print(data)