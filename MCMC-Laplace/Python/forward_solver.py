import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import time

###########################################################################
############ list of "exact" measurement values, z_hat ####################
###########################################################################

z_hat = np.array(
     [0.06076511762259369, 0.09601910120848481,
      0.1238852517838584,  0.1495184117375201,
      0.1841596127549784,  0.2174525028261122,
      0.2250996160898698,  0.2197954769002993,
      0.2074695698370926,  0.1889996477663016,
      0.1632722532153726,  0.1276782480038186,
      0.07711845915789312, 0.09601910120848552,
      0.2000589533367983,  0.3385592591951766,
      0.3934300024647806,  0.4040223892461541,
      0.4122329537843092,  0.4100480091545554,
      0.3949151637189968,  0.3697873264791232,
      0.33401826235924,    0.2850397806663382,
      0.2184260032478671,  0.1271121156350957,
      0.1238852517838611,  0.3385592591951819,
      0.7119285162766475,  0.8175712861756428,
      0.6836254116578105,  0.5779452419831157,
      0.5555615956136897,  0.5285181561736719,
      0.491439702849224,   0.4409367494853282,
      0.3730060082060772,  0.2821694983395214,
      0.1610176733857739,  0.1495184117375257,
      0.3934300024647929,  0.8175712861756562,
      0.9439154625527653,  0.8015904115095128,
      0.6859683749254024,  0.6561235366960599,
      0.6213197201867315,  0.5753611315000049,
      0.5140091754526823,  0.4325325506354165,
      0.3248315148915482,  0.1834600412730086,
      0.1841596127549917,  0.4040223892461832,
      0.6836254116578439,  0.8015904115095396,
      0.7870119561144977,  0.7373108331395808,
      0.7116558878070463,  0.6745179049094283,
      0.6235300574156917,  0.5559332704045935,
      0.4670304994474178,  0.3499809143811,
      0.19688263746294,    0.2174525028261253,
      0.4122329537843404,  0.5779452419831566,
      0.6859683749254372,  0.7373108331396063,
      0.7458811983178246,  0.7278968022406559,
      0.6904793535357751,  0.6369176452710288,
      0.5677443693743215,  0.4784738764865867,
      0.3602190632823262,  0.2031792054737325,
      0.2250996160898818,  0.4100480091545787,
      0.5555615956137137,  0.6561235366960938,
      0.7116558878070715,  0.727896802240657,
      0.7121928678670187,  0.6712187391428729,
      0.6139157775591492,  0.5478251665295381,
      0.4677122687599031,  0.3587654911000848,
      0.2050734291675918,  0.2197954769003094,
      0.3949151637190157,  0.5285181561736911,
      0.6213197201867471,  0.6745179049094407,
      0.690479353535786,   0.6712187391428787,
      0.6178408289359514,  0.5453605027237883,
      0.489575966490909,   0.4341716881061278,
      0.3534389974779456,  0.2083227496961347,
      0.207469569837099,   0.3697873264791366,
      0.4914397028492412,  0.5753611315000203,
      0.6235300574157017,  0.6369176452710497,
      0.6139157775591579,  0.5453605027237935,
      0.4336604929612851,  0.4109641743019312,
      0.3881864790111245,  0.3642640090182592,
      0.2179599909280145,  0.1889996477663011,
      0.3340182623592461,  0.4409367494853381,
      0.5140091754526943,  0.5559332704045969,
      0.5677443693743304,  0.5478251665295453,
      0.4895759664908982,  0.4109641743019171,
      0.395727260284338,   0.3778949322004734,
      0.3596268271857124,  0.2191250268948948,
      0.1632722532153683,  0.2850397806663325,
      0.373006008206081,   0.4325325506354207,
      0.4670304994474315,  0.4784738764866023,
      0.4677122687599041,  0.4341716881061055,
      0.388186479011099,   0.3778949322004602,
      0.3633362567187364,  0.3464457261905399,
      0.2096362321365655,  0.1276782480038148,
      0.2184260032478634,  0.2821694983395252,
      0.3248315148915535,  0.3499809143811097,
      0.3602190632823333,  0.3587654911000799,
      0.3534389974779268,  0.3642640090182283,
      0.35962682718569,    0.3464457261905295,
      0.3260728953424643,  0.180670595355394,
      0.07711845915789244, 0.1271121156350963,
      0.1610176733857757,  0.1834600412730144,
      0.1968826374629443,  0.2031792054737354,
      0.2050734291675885,  0.2083227496961245,
      0.2179599909279998,  0.2191250268948822,
      0.2096362321365551,  0.1806705953553887,
      0.1067965550010013])


###########################################################################
####### do all precomputations necessary for MCMC simulations #############
###########################################################################

# Define the mesh width
h = 1/32

# Define characteristic function of unit square
def heaviside(x) :
    if x<0 :
        return 0
    else :
        return 1
    
def S(x,y) :
    return heaviside(x)*heaviside(y) * (1-heaviside(x-h))*(1-heaviside(y-h));

# Define tent function on the domain [0,2h]x[0,2h]
def phi(x,y) :
    return ((x+h)*(y+h)*S(x+h,y+h) + (h-x)*(h-y)*S(x,y) 
            + (x+h)*(h-y)*S(x+h,y) + (h-x)*(y+h)*S(x,y+h))/h**2

# Define conversion function for dof's from 2D to scalar label, and
# its inverse
def ij_to_dof_index(i,j) :
    return 33*j+i

def inv_ij_to_dof_index(k) :
    return [k-33*int(k/33),int(k/33)]


# Construct measurement matrix, M, for measurements
xs = np.arange(1./14,13./14,1./14);    #measurement points

M = np.zeros((13,13,33**2));
for k in range(33**2) :
    c = inv_ij_to_dof_index(k)
    for i in range(13) :
        for j in range(13) :
            M[i,j,k] = phi(xs[i]-h*c[0], xs[j]-h*c[1])
M = M.reshape((13**2, 33**2))
M = scipy.sparse.csr_matrix(M);

# Construct local overlap matrix, A_loc, and identity matrix Id
A_loc = np.array([[2./3,  -1./6,  -1./3,  -1./6],
                  [-1./6,  2./3,  -1./6,  -1./3],
                  [-1./3, -1./6,   2./3,  -1./6],
                  [-1./6, -1./3,  -1./6,   2./3]])
Id = np.eye(33**2,33**2)

# Locate boundary labels
boundaries = ([ij_to_dof_index(i,0) for i in range(33)] +
              [ij_to_dof_index(i,32) for i in range(33)] +
              [ij_to_dof_index(0,j+1) for j in range(31)] +
              [ij_to_dof_index(32,j+1) for j in range(31)])

# Define RHS of FEM linear system, AU = b
b = np.ones(33**2)*10*h**2
b[boundaries] = 0    #enforce boundary conditions on b





###########################################################################
###################### forward solver function ############################
###########################################################################

def forward_solver(theta) :
    # Initialize matrix A for FEM linear solve, AU = b
    A = np.zeros((33**2,33**2))

    # Build A by summing over contribution from each cell
    for i in range(32) :
        for j in range (32) :
            # Find local coefficient in 8x8 grid
            thet = theta[int(i/4)+int(j/4)*8]

            # Update A by including contribution from cell (i,j)
            dof = [ij_to_dof_index(i,j),
                   ij_to_dof_index(i,j+1),
                   ij_to_dof_index(i+1,j+1),
                   ij_to_dof_index(i+1,j)]
            A[np.ix_(dof,dof)] += thet*A_loc

    # Enforce boundary condition: Zero out rows and columns, then
    # put a one back into the diagonal entries.
    A[boundaries,:] = 0
    A[:,boundaries] = 0
    A[boundaries,boundaries] = 1

    # Solve linear equation for coefficients, U, and then
    # get the Z vector by multiplying by the measurement matrix
    u = spsolve(scipy.sparse.csr_matrix(A), b)
    
    z = M * u
    
    return z




###########################################################################
################# compute log probability, log pi #########################
###########################################################################

def log_likelihood(theta) :
    z = forward_solver(theta)
    misfit = z - z_hat
    sig = 0.05             #likelihood standard deviation
    return -np.dot(misfit,misfit)/(2*sig**2)

def log_prior(theta) :
    sig_pr = 2             #prior (log) standard deviation
    return -np.linalg.norm(np.log(theta))**2/(2*sig_pr**2)

def log_posterior(theta) :
    return log_likelihood(theta) + log_prior(theta)



###########################################################################
############# A function to test against known output #####################
###########################################################################


def verify_against_stored_tests() :
    for i in range(10) :
        print ("Verifying against data set", i)

        # Read the input vector
        f_input = open ("../testing/input.{}.txt".format(i), 'r')
        theta = np.fromfile(f_input, count=64, sep=" ")

        # Then compute both the forward solution and its statistics.
        # This is not efficiently written here (it calls the forward
        # solver twice), but we don't care about efficiency here since
        # we are only computing with ten samples
        this_z              = forward_solver(theta)
        this_log_likelihood = log_likelihood(theta)
        this_log_prior      = log_prior(theta)

        # Then also read the reference output generated by the C++ program:
        f_output_z = open ("../testing/output.{}.z.txt".format(i), 'r')
        f_output_likelihood = open ("../testing/output.{}.likelihood.txt".format(i), 'r')
        f_output_prior = open ("../testing/output.{}.prior.txt".format(i), 'r')

        reference_z              = np.fromfile(f_output_z, count=13**2, sep=" ")
        reference_log_likelihood = float(f_output_likelihood.read())
        reference_log_prior = float(f_output_prior.read())

        print ("  || z-z_ref ||  : ",
               np.linalg.norm(this_z - reference_z))
        print ("  log likelihood : ",
               "Python value=", this_log_likelihood,
               "(C++ reference value=", reference_log_likelihood,
               ", error=", abs(this_log_likelihood - reference_log_likelihood),
               ")")
        print ("  log prior      : ",
               "Python value=", this_log_prior,
               "(C++ reference value=", reference_log_prior,
               ", error=", abs(this_log_prior - reference_log_prior),
               ")")


def time_forward_solver() :
    begin = time.time()

    n_runs = 100
    for i in range(n_runs) :
        # Create a random vector (with entries between 0 and 1), scale
        # it by a factor of 4, subtract 2, then take the exponential
        # of each entry to get random entries between e^{-2} and
        # e^{+2}
        theta = np.exp(np.random.rand(64) * 4 - 2)
        z = forward_solver(theta)
    end = time.time()
    print ("Time per forward evaluation:", (end-begin)/n_runs)

verify_against_stored_tests()
time_forward_solver()
