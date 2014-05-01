#include "Matrix.hpp"
// This function will take a matrix A, which has been multiplied the time step "dt",
// which hence forth called Adt.
// The formula is basically (I + Adt + 1/2 * Adt^2 + 1/3! * Adt^3 + ...   )
matrix Exp_A( matrix Adt )
{
    int k, m;
    double factorial, err;
    matrix A, 
		   ArK, 
		   Tmp;  // Workspace variables. 

    // Check that Adt is square.
    if( Adt.high() != Adt.wide() )
        return A; //return empty matrix as sign of an error.

    // Allocate space for output matrix.
    A = matrix( Adt.high(), Adt.wide() );

    if( A.isValid() )  // Check for valid allocation of A
    {
        // Load diagonals with ones.
        for( k = 0; k < Adt.high(); k++ ) 
           for( m = 0; m < Adt.wide(); m++)
			  A(k,m) = (k==m) ? 1.0 : 0.0;

        // Make a copy of Adt.
        ArK = Adt;

        // Check for ArK valid.
        if( ArK.isInvalid() )
        {
            return ArK;
        }

        // Initialize n!, error and index.
        factorial = 1.0;
        err = 1.0;
        k = 2;
    
	    // Loop while err is non-zero, and the matrices are still valid.
        while( err > 0.0 && A.isValid() && ArK.isValid() )
        {
            // Add in next term in exp( A * dt )
            Tmp = A + ArK; // on First pass it, Tmp is allocated.

			if( Tmp.isInvalid() )
			{
				break; // Exit loop and finish up.
			}
            // Measure change from last term.
            err = 0.0;                    
            for( m = 0; m < A.high()*A.wide(); m++ )
            {   // Compute difference 
				err += ( A(m) - Tmp(m) )
                      * ( A(m) - Tmp(m) );   
				// Then copy over the next values for A.
				A(m) = Tmp(m);
			}
			// Compute Next A raised to K
            Tmp = ArK * Adt;  
            // Divided by N!
            ArK = Tmp * (1.0 / (double)k );  
            
            k++; // Next k.
        } // End of Loop until error small.
    	
    } // End of Check for valid allocation of A

	 return A;

} // End of Exp_A()
