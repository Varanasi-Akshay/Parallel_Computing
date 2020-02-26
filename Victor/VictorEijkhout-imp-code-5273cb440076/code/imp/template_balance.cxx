/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** template_balance.cxx : 
 **** mode-independent template for load balancing
 ****
 ****************************************************************/

/*! \page balance Load balancing

  This is incomplete.
*/

#include "template_common_header.h"
#include "balance_functions.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Balance options:\n");
    printf("  -B     : do NOT load balance\n");
    printf("  -n nnn : local number of points\n");
    printf("  -s nnn : number of steps\n");
    printf("  -trace : print norms\n");
    printf(":\n");
  };
  
  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("balance");
  
  /* Print help information if the user specified "-h" argument */
  if (env->has_argument("h")) {
    printf("Usage: %s [-d] [-s nsteps] [-n size] [ -B ]\n",argv[0]);
    return -1;
  }
      
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
  processor_coordinate mycoord( std::vector<int>{mytid} );
#endif
  int ntids = arch->nprocs();
  IMP_decomposition* decomp = new IMP_decomposition(arch);

  int
    laststep = env->iargument("s",3),
    localsize = env->iargument("n",10),
    trace = env->has_argument("trace"),
    balance = ! env->has_argument("B");

  distribution
    *block = new IMP_block_distribution(decomp,localsize,-1),
    *load = new IMP_replicated_distribution(decomp,ntids);
  block->set_name("block distribution");
  load->set_name("local distribution");

  domain_coordinate global_last = block->global_last_index()+1;
  auto stretch_to =
    new distribution_stretch_operator(global_last);

  // input and output vector are permanent
  auto
    input_vector = std::shared_ptr<object>( new IMP_object(block) ),
    output_vector = std::shared_ptr<object>( new IMP_object(block) );
  std::shared_ptr<object> tmp_vector;

  
  for (int istep=0; istep<=laststep; istep++) {

    try {
      /*
       * The work algorithm
       */
      algorithm *onestep = new IMP_algorithm(decomp);
      onestep->set_kernel_zero(istep-1);

      // we need to accept the input vector, whatever it was before
      onestep->add_kernel( new IMP_origin_kernel(input_vector) );

      // this is the work step
      auto work = new IMP_copy_kernel(input_vector,output_vector);
      if (balance) {
	work->set_localexecutefn
	  ( std::function< kernel_function_proto >{
	    [laststep] ( kernel_function_args ) -> void {
	      setmovingweight( kernel_function_call , laststep );
	    } } );
      } else {
	work->set_localexecutefn( &vecsetconstantone );
      }
      onestep->add_kernel(work);

      // get runtime statistics
      auto stats_vector = std::shared_ptr<object>( new IMP_object(load) );
      onestep->add_kernel( new IMP_stats_kernel(output_vector,stats_vector,summing) );

      // preserve output for the next loop
      if (istep<laststep) {
	tmp_vector = std::shared_ptr<object>( new IMP_object(block) );
	onestep->add_kernel( new IMP_copy_kernel(output_vector,tmp_vector) );
      }
      try {
	onestep->analyze_dependencies();
	onestep->execute();
      } catch (std::string c) {
	throw(fmt::format("Step {} : onestep error {}\n",istep,c));
      }
      if (trace && mytid==0) {
	double *stats_data = stats_vector->get_data(mycoord), maxwork = 0;
	for (int p=0; p<ntids; p++) if (stats_data[p]>maxwork) maxwork = stats_data[p];
	fmt::print("Step {}: {}\nStats: {}; max={}\n",
		   istep,block->as_string(),
		   stats_vector->values_as_string(mycoord),maxwork
		   );
      }

      if (istep<laststep) {
	double *stats_data = stats_vector->get_data(mycoord);
	//snippet apply_average
	auto average =
	  new distribution_sigma_operator
	  ( [stats_data] (distribution *d) -> distribution* {
	    try {
	      return transform_by_average(d,stats_data);
	    } catch (std::string c) {
	      throw(fmt::format("Error in averaging: {}",c));
	    } } );
	//snippet end

	/*
	 * Load balancing for the next iteration
	 * the compute output is in tmp_vector, which is of distribution `block'
	 */
	// compute new distribution and reallocate input/output
	try {
	  distribution *avgblock;

	  try { avgblock = block->operate(average);
	  } catch (std::string c) { throw(fmt::format("Operate average: {}",c)); };

	  try { block = avgblock->operate(stretch_to);
	  } catch (std::string c) { throw(fmt::format("Operate stretch: {}",c)); };

	} catch (std::string c) {
	  throw(fmt::format("Step {}: trouble applying average: {}",istep,c));
	} catch (...) {
	  throw(fmt::format("Step {}: unknown trouble applying average",istep));
	}
	input_vector = std::shared_ptr<object>( new IMP_object(block) );
	input_vector->set_name(fmt::format("in{}",istep+1));
	output_vector = std::shared_ptr<object>( new IMP_object(block) );
	output_vector->set_name(fmt::format("out{}",istep+1));

	algorithm *load_balance = new IMP_algorithm(decomp);
	load_balance->add_kernel( new IMP_origin_kernel(tmp_vector) );
	load_balance->add_kernel( new IMP_copy_kernel(tmp_vector,input_vector) );

	try {
	  load_balance->analyze_dependencies();
	  load_balance->execute();
	} catch (std::string c) { fmt::print("Balance step {}: {}\n",istep,c); }
      }
    } catch (std::string c) { fmt::print("Strange error in step {}: {}\n",istep,c);
    } catch (std::logic_error e) { fmt::print("Logic error: {}\n",e.what()); }
  }

  delete env;

  return 0;
}

#if 0
  for (int iter=0; iter<50; iter++) {
    //snippet apply_average
    //snippet end
  }
      // if (mytid==0)
      // 	fmt::print("Step {}: block is {}\n",istep,block->as_string());
      // fmt::print("{} Step {} output : {}\n",
      // 		 mycoord.as_string(),istep,output_vector->values_as_string(mycoord));
#endif

