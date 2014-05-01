#include <stdio.h>
#include <math.h>
#include <complex>
//local include files
#include "MatrixOutputs.hpp"
#include "Exp_A.hpp"

using namespace std;

#define eps 2.22e-16

void main( void )
{
	int k;
	//char array for file name
	char Name[ 64 ];
	matrix A, exp_Adt, x( 3 ), b( 3 );
	double dt, // Time Step
		TimeInterval = 6, // Time interval for simulationl
		gamma = 2.0e-2,
		epsilon = 2.0e-2,
		Tg = 3.0,
		Ts = 0.5,
		kp,
		ki,
		alpha = Tg*Ts,
		beta = Tg + Ts;
	FILE *fout;
	// Allocate and set up system matrix.
	A = matrix( 3, 3 );
	// Check for valid allocation of A
	if ( A.isInvalid( ) )
	{
		printf( "Error allocating A\n" );
		getchar( );
		return;
	} // end of check for valid allocation of A
	// Load state variable matrix.
	for ( ki = 0; ki <= 4; ki += 2 )
	{
		for ( kp = 4; kp <= 12; kp += 4 )
		{
			gamma = 1.0 + kp;
			A( 0, 0 ) = 0.0;
			A( 0, 1 ) = 1.0;
			A( 0, 2 ) = 0.0;
			A( 1, 0 ) = 0.0;
			A( 1, 1 ) = 0.0;
			A( 1, 2 ) = 1.0;
			A( 2, 0 ) = -ki / alpha;
			A( 2, 1 ) = -gamma / alpha;
			A( 2, 2 ) = -beta / alpha;
			// We will use the "exact" solution, exp( A * dt );
			// Note that the time step is pretty large.
			dt = 1.0e-3;
			exp_Adt = Exp_A( A * dt );

			// Check for A valid.
			if ( exp_Adt.isInvalid( ) )
			{
				printf( "Error creating Exp_A for exact Solution\n" );
				getchar( );
				return;
			}
			printf( "Print out of matrix exp( A * dt )\n\n" );
			PrintMatrix( exp_Adt );

			// Having computed exp( A * dt ),
			// We now use it to simulate system.
			// Open file for output data, check for validity.
			//creates files with different names based on the ki and kp values
			sprintf( Name, "ki_%i_and_kp_%i.csv", (int)ki, (int)kp );
			fout = fopen( Name, "w" );

			if ( fout )
			{
				x( 0 ) = 0.0; // initial values are zero.
				x( 1 ) = 0.0;
				x( 2 ) = 0.0;
				// Input vector
				b( 0 ) = 0.0;
				b( 1 ) = 0.0;
				b( 2 ) = ( kp + ki*Ts*dt ) / alpha; // This is the value of the inputs
				// at the start of the step.
				// Loop through time interval
				for ( k = 0.0; k < (int)( TimeInterval / dt ); k++ )
				{
					fprintf( fout, "%18.16lg, %18.16lg, %18.16lg, %18.16lg\n",
						k*dt, x( 0 ), x( 1 ), x( 2 ) );
					// Take step by A*x
					x = exp_Adt * x + b;
					if ( !k )
						b( 2 ) = ki*dt / alpha;
				} // end of time step loop.
				fclose( fout );
			} // end of valid file test for Exact Solution.
		}
	}
	printf( "\n Simulation complete:" );
	getchar( );
	return;
} // end of main
