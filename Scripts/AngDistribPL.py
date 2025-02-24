"""
Copyright (c) 2025 VOLCHOK Evgeniia
for contacts e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

import plotly.graph_objects as go
import numpy as np
from math import pi

data={}
data['ith'],data['iphi'],data['Int']=np.loadtxt('./Scripts/Data/ResPowerAlpha2.5.dat', unpack=True,usecols=[0,2,10],comments='#',)

R=np.zeros(shape=(301,601))

for i in range (0, len(data['Int'])):
	ith=data['ith'][i]
	iphi=data['iphi'][i]
	R[int(ith)][int(iphi)]=data['Int'][i]
	print(i)
	

theta=np.linspace(0, pi, 301)
phi=np.linspace(0, 2.*pi, 601)
PHI, THETA = np.meshgrid(phi, theta)

x=R*np.cos(PHI)*np.sin(THETA)
y=R*np.sin(PHI)*np.sin(THETA)
z=R*np.cos(THETA)


print('plotting')

surface=go.Surface(x=x, y=y, z=z)

fig = go.Figure(data=[surface])
fig.update_layout()

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.3, y=0., z=1.8)
)

f1 = dict(
    family="Old Standard TT, serif",
    size=24,
    color="black"
)
f2 = dict(
    family="Old Standard TT, serif",
    size=18,
    color="black"
)

fig.update_layout(
    width=1200,
    height=1000,
    margin=dict(l=10, r=10, b=10, t=10),
    scene=dict(
        xaxis=dict(
            title=dict(text="x", font=f1),
            tickfont=f2
        ),
        yaxis=dict(
            title=dict(text="y", font=f1),
            tickfont=f2
        ),
        zaxis=dict(
            title=dict(text="z", font=f1),
            tickfont=f2
        ),
        bgcolor='white'
    ),
    scene_camera=camera,
    coloraxis_colorbar=dict(
        title=dict(text="", font=f1),
        tickfont=f2
    )
)

#fig.write_image("./png5/fig1.png")
fig.show()


