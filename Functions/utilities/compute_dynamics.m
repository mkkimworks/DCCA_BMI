function DN                 = compute_dynamics(XF)

Xdot                        = gradient(XF);
dim                         = size(XF,2);
X_                          = [];
for n = 1 : dim
    X_                      = blkdiag(X_, XF);
end

x_dot                       = Xdot(:);
m_star                      = (x_dot'*x_dot)^-1*x_dot' * X_;


M_star                      = reshape(m_star', dim, dim);
M_star_skew                 = (M_star - M_star')/2;
dynamics                    = XF * M_star_skew;

DN.M                        = M_star_skew;
DN.dynamics                 = dynamics;