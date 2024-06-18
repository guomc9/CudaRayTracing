# CudaRayTracing
## Ray Tracing
### Rendering Equation
$$L_o(p, \omega_o) = L_e(p, \omega_o) + \int_\Omega f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (n · \omega_i) \mathrm{d}\omega_i$$
### Monte Carlo Integration
$$\int_Ω f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (n \cdot \omega_i) d\omega_i \approx\frac{1}{N} \sum_{k=1}^{N} \frac{f_r(p, \omega_{i,k}, \omega_o) L_i(p, \omega_{i,k}) (n \cdot \omega_{i,k})}{p(\omega_{i,k})}$$
* $`\omega_{i,k}`$ : Incident solid angle from random sphere sampling 
* $`p(\omega_{i,k})`$ : $`\frac{1}{2\pi}`$

### Direct and Indirect Radiance
$$L_o(p, \omega_o) \approx L_e(p, \omega_o) + L_{direct}(p, \omega_o) + L_{indirect}(p, \omega_o)$$


$$L_{direct}(p, \omega_o) = \int_\Omega f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (n \cdot \omega_i) \mathrm{d}\omega_i$$
$$L_{direct}(p, \omega_o) =\int_A f_r(p, \omega_i, \omega_o) L_i(p_i, \omega_i)(n \cdot \omega_i)\frac{(n'\cdot \omega_i)}{\Vert x-x'\Vert_2^2}\mathrm{d}A$$

$$L_{direct}(p, \omega_o) \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f_r(p, \omega_{i,k}, \omega_o) L_i(p, \omega_{i,k}) (n \cdot \omega_{i,k})(n_k'\cdot\omega_{i,k})}{p(\omega_{i,k})\Vert x-x_k'\Vert_2^2}$$


* $`A`$ : Area of light area
* $`\omega_{i,k}`$ : Incident solid angle from random light area sampling
* $`p(\omega_{i,k})`$ : $`\frac{1}{A}`$

### Specular Material
In sphere coordinate, sphere area formula:
$$A={\int\int}_{\Omega} r^2 \sin\theta\mathrm{d}\theta\mathrm{d\phi}$$
For area of unit sphere, $`\theta`$ from $`0`$ to $`\Delta\theta`$, $`\phi`$ from $`0`$ to $`\Delta\phi`$:

$$S=\int_{0}^{\Delta\theta}\int_{0}^{\Delta\phi}\sin\theta\mathrm{d}\theta\mathrm{d\phi}=(1-\cos\Delta\theta)\Delta\phi$$

Specular material radiance:


$$L_{specular}=\int_\Omega f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (n \cdot \omega_i) d\omega_i \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f_r(p, \omega_{i,k}, \omega_o) L_i(p, \omega_{i,k}) (n \cdot \omega_{i,k})}{p(\omega_{i,k})}$$

* $`\omega_{i,k}`$ : Incident solid angle from random sphere sampling of $`\{(\theta, \phi)| \theta_0-\frac{\Delta\theta}{2}\le \theta \le \theta_0+\frac{\Delta\theta}{2}, \phi_0-\frac{\Delta\phi}{2}\le \phi \le \phi_0+\frac{\Delta\phi}{2}\}$, $\omega_i=(\theta_0, \phi_0)$, $\Delta\theta\in(0, \pi)$, $\Delta\phi\in(0, \pi)`$
* $`p(\omega_{i,k})`$ : $`\frac{1}{S}=\frac{1}{(1-\cos\Delta\theta)\Delta\phi}`$

From Cartesian coordinate system to Sphere coordinate system:

$$r = \sqrt{x^2 + y^2 + z^2}$$

$$\theta = \arccos\left(\frac{z}{r}\right)$$

$$\phi = \arctan\left(\frac{y}{x}\right) \ \ \  \text{for } x > 0$$

$$\phi = \arctan\left(\frac{y}{x}\right) + \pi \ \ \  \text{for } x < 0 \text{ and } y \geq 0$$

$$\phi = \arctan\left(\frac{y}{x}\right) - \pi \ \ \  \text{for } x < 0 \text{ and } y < 0 $$

$$\phi = \frac{\pi}{2} \ \ \  \text{for } x = 0 \text{ and } y > 0 $$

$$\phi = -\frac{\pi}{2} \ \ \  \text{for } x = 0 \text{ and } y < 0 $$

$$\phi = \text{undefined} \ \ \  \text{for } x = 0 \text{ and } y = 0 $$

From Sphere coordinate system to Cartesian coordinate system:
$$x=r\sin\theta\cos\phi$$

$$y=r\sin\theta\sin\phi$$

$$z=r\cos\theta$$
