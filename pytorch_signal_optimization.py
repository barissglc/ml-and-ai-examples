import numpy as np
import matplotlib.pyplot as plt
sinyal_uzunlugu = 200

# sinyalimizi olusturalim

t = np.linspace(0,2 * np.pi, sinyal_uzunlugu)
sinyal = np.sin(t) + np.random.random(sinyal_uzunlugu,) * 0.1

# sinyalimizi gorsellestirelim

plt.plot(sinyal)
plt.title("Sinyal")
plt.show()

#rastgele gürültümüz olsun turuncu renkte
gurultu = np.random.random(sinyal_uzunlugu,) * 2 - 1

plt.plot(gurultu, color="orange")
plt.title("Gürültü")
plt.show()

#beraber görelim aynı grafikte farklı renklerde

plt.plot(sinyal, label="Sinyal")
plt.plot(gurultu, label="Gürültü")
plt.legend()
plt.title("Sinyal ve Gürültü")
plt.show()

#torch

import torch

hedef_tensor = torch.from_numpy(sinyal)
baslangic_tensor = torch.from_numpy(gurultu)

baslangic_variable = torch.autograd.Variable(baslangic_tensor, requires_grad=True)
hedef_variable = torch.autograd.Variable(hedef_tensor, requires_grad=False)

lr_ = 0.05
optimizer = torch.optim.Adam([baslangic_variable], lr=lr_)

loss_fn = torch.nn.MSELoss()

import matplotlib.animation as animation

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150  
plt.ioff()
fig, ax = plt.subplots()



def animate(update_no):
    optimizer.zero_grad()

    # Hata hesaplama ve optimizasyon
    loss = loss_fn(hedef_variable, baslangic_variable)
    loss.backward()
    optimizer.step()

    
    ax.clear()
    ax.plot(hedef_variable.detach().numpy(), label="Hedef")
    ax.plot(baslangic_variable.detach().numpy(), label="Başlangıç")
    ax.set_title(f"Update: {update_no}, Loss: {loss.item():.4f}")
    ax.legend()



anim = animation.FuncAnimation(fig, animate, frames=50)

# Animasyonu göster
plt.show()