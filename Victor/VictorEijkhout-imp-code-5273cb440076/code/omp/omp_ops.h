// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** omp_ops.h: Header file for the OpenMP operation kernels
 ****
 ****************************************************************/

#ifndef OMP_OPS_H
#define OMP_OPS_H 1

#include "omp_base.h"
#include "imp_functions.h"
#include "imp_ops.h"

/*! \page ompops OMP Operations
  The basic mechanism of declaring distributions, objects, and kernels
  is powerful enough by itself. However, for convenience a number 
  of common operations have been declared.

  File omp_ops.h contains:
  - \ref omp_spmvp_kernel : the distributed sparse matrix vector product
  - \ref omp_innerproduct : the distributed inner product
  - \ref omp_outerproduct_kernel : combine distributed data with replicated data
*/


/*!
  Copying is simple
*/
class omp_copy_kernel : public omp_kernel,public copy_kernel {
public:
  omp_copy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : copy_kernel(in,out),omp_kernel(in,out),kernel(in,out),
      entity(entity_cookie::KERNEL) {};
};

/*!
  Scale a vector by a scalar.
*/
class omp_scale_kernel : public omp_kernel,public scale_kernel {
public:
  omp_scale_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out),omp_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
  omp_scale_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out),omp_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  Scale a vector down by a scalar.
*/
class omp_scaledown_kernel : public omp_kernel,public scaledown_kernel {
public:
  omp_scaledown_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),omp_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
  omp_scaledown_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),omp_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  AXPY is a local kernel; the scalar is passed as the context. The scalar
  comes in as double*, this leaves open the possibility of an array of scalars,
  also it puts my mind at ease re that casting to void*.
*/
class omp_axpy_kernel : public omp_kernel,public axpy_kernel {
public:
  omp_axpy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,double *x )
    : axpy_kernel(in,out,x),omp_kernel(in,out),kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  Add two vectors together. For a more general version see \ref omp_axbyz_kernel.
*/
class omp_sum_kernel : public omp_kernel,public sum_kernel {
public:
  omp_sum_kernel( std::shared_ptr<object> in1,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),sum_kernel(in1,in2,out),omp_kernel(in1,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  AXBYZ is a local kernel; the scalar is passed as the context. The scalar
  comes in as double*, this leaves open the possibility of an array of scalars,
  also it puts my mind at ease re that casting to void*.
*/
class omp_axbyz_kernel : public omp_kernel, public axbyz_kernel {
protected:
public:
  omp_axbyz_kernel( char op1,std::shared_ptr<object> s1,std::shared_ptr<object> x1,
		    char op2,std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out )
    : kernel(x1,out),omp_kernel(x1,out),axbyz_kernel(op1,s1,x1,op2,s2,x2,out),
      entity(entity_cookie::KERNEL)  {};
  //! Abbreviated creation leaving the scalars untouched
  omp_axbyz_kernel( std::shared_ptr<object> s1,std::shared_ptr<object> x1, std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out ) :
    omp_axbyz_kernel( '+',s1,x1, '+',s2,x2, out ) {};
};

/*!
  A whole class for operations between replicated scalars
*/
class omp_scalar_kernel : public omp_kernel,public scalar_kernel {
protected:
public:
  omp_scalar_kernel( std::shared_ptr<object> in1,const char *op,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : scalar_kernel(in1,op,in2,out),omp_kernel(in1,out),kernel(in1,out),
      entity(entity_cookie::KERNEL)  {};
};

/*! A sparse matrix-vector product is easily defined
  from the local product routine and using the 
  same sparse matrix as the index pattern of the beta distribution
*/
class omp_spmvp_kernel : virtual public omp_kernel {
public:
  omp_spmvp_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,sparse_matrix *mat)
    : kernel(in,out),omp_kernel(in,out),
      entity(entity_cookie::KERNEL)  {
    set_name("sparse-mvp");
    dependency *d = last_dependency();
    d->set_index_pattern( mat );
    d->set_name(fmt::format("spmvp-into-{}",out->get_name()));

    set_localexecutefn
      ( [mat] ( kernel_function_args ) -> void {
	return local_sparse_matrix_vector_multiply( kernel_function_call,(void*)mat ); } );
  };
};

class omp_centerofmass_kernel : virtual public omp_kernel {
public:
  omp_centerofmass_kernel(std::shared_ptr<object> bot,std::shared_ptr<object> top)
    : omp_kernel(bot,top),kernel(bot,top),
      entity(entity_cookie::KERNEL)  {
    set_localexecutefn( &scansum );
    last_dependency()->set_signature_function_function
      ( [] (index_int i) -> std::shared_ptr<indexstruct> {
	return doubleinterval(i); } );
  };
};

class omp_sidewaysdown_kernel : virtual public omp_kernel {
private:
  distribution *level_dist,*half_dist;
  std::shared_ptr<object> expanded,multiplied;
  omp_kernel *expand,*multiply,*sum;
public:
  omp_sidewaysdown_kernel( std::shared_ptr<object> top,std::shared_ptr<object> side,std::shared_ptr<object> out,omp_sparse_matrix *mat )
    : omp_kernel(top,out),kernel(top,out),
      entity(entity_cookie::KERNEL)  {
    level_dist = dynamic_cast<distribution*>(out.get());
    half_dist = dynamic_cast<distribution*>(top.get());

    expanded = std::shared_ptr<object>( new omp_object(level_dist) );
    expand = new omp_kernel(top,expanded);
    expand->set_localexecutefn( &scanexpand );
    expand->last_dependency()->set_signature_function_function
      ( [] (index_int i) -> std::shared_ptr<indexstruct> {
	return halfinterval(i); } );
    expanded->set_name(fmt::format("parent-expanded-{}",out->get_object_number()));

    multiplied = std::shared_ptr<object>( new omp_object(level_dist) );
    multiply = new omp_spmvp_kernel(side,multiplied,mat);
    multiplied->set_name(fmt::format("cousin-product-{}",out->get_object_number()));
    
    sum = new omp_sum_kernel(expanded,multiplied,out);
    sum->set_name("sidewaysdown-sum");
  };
  ~omp_sidewaysdown_kernel() { delete expand; delete multiply; delete sum;
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(expand,multiply,sum);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(expand,multiply,sum);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() {
    expand->execute(); multiply->execute(); sum->execute();
  };
  std::string as_string() override {
    fmt::MemoryWriter w;
    w.write("K[{}]=\n",get_name());
    w.write("  <<{}>>\n",expand->as_string());
    w.write("  <<{}>>\n",multiply->as_string());
    w.write("  <<{}>>",sum->as_string());
    return w.str();
  };
};

/*!
  An OpenMP reduction is way easier than an MPI one.
  I doubt that the two can be unified, but maybe this can be the base reduction class
*/
class omp_reduction_kernel : virtual public omp_kernel {
private: // we need to keep them just to destroy them
protected:
public:
  omp_reduction_kernel( std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum)
    : omp_kernel(local_value,global_sum),kernel(local_value,global_sum),
      entity(entity_cookie::KERNEL)  {
    if (!global_sum->has_type_replicated())
      throw("reduction sum needs to be replicated\n");

    set_name("scalar reduction");
    set_explicit_beta_distribution( new omp_gathered_distribution(global_sum.get()) );
    set_localexecutefn( &summing );

  };
};


//snippet innerproductkernel
/*!
  An inner product is somewhat tricky. While it behaves as a single kernel,
  it is composed of two kernels: one gather followed by a local summing.
  This means we have to overwrite #analyze_dependencies and #execute
  to call the corresponding routines of the two enclosed kernels.

  \todo add an option for an ortho parameter
  \todo test that "global_sum" is replicated_scalar
*/
#if 1
class omp_innerproduct_kernel : public omp_kernel,public innerproduct_kernel {
public:
  omp_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),omp_kernel(v1,global_sum),
      innerproduct_kernel(v1,v2,global_sum),
      entity(entity_cookie::KERNEL) {};
};
#else
class omp_innerproduct_kernel : virtual public omp_kernel {
private:
  distribution *local_scalar,*gathered_scalar; // we need to keep them just to destroy them
  std::shared_ptr<object> local_value;
protected:
  omp_kernel *prekernel,*sumkernel;
public:
  omp_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : omp_kernel(v1,global_sum),kernel(v1,global_sum),
      entity(entity_cookie::KERNEL)  {
    decomposition *decomp = global_sum;
    set_name("inner-product");

    // intermediate object for local sum:
    local_scalar = new omp_scalar_distribution(decomp); //(v2,v2->global_ndomains());
    local_value = std::shared_ptr<object>( new omp_object(local_scalar) );
    local_value->set_name("local-inprod-value");
    
    // local inner product kernel
    dependency *d;
    prekernel = new omp_kernel(v1,local_value);
    prekernel->set_name("local-innerproduct");
    // v1 is in place
    d = prekernel->last_dependency();
    d->set_explicit_beta_distribution(v1.get());
    d->set_name("inprod wait for in vector");
    // v2 is in place
    prekernel->add_in_object(v2);
    d = prekernel->last_dependency();
    d->set_explicit_beta_distribution(v2.get());
    d->set_name("inprod wait for second vector");
    // function
    prekernel->set_localexecutefn( &local_inner_product );

    // reduction kernel
    sumkernel = new omp_reduction_kernel(local_value,global_sum);
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(prekernel,sumkernel);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(prekernel,sumkernel);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() { // synckernel->execute();
    prekernel->execute(); sumkernel->execute(); };
  kernel *get_prekernel() { return prekernel; };
};
//snippet end
#endif

/*!
  A vector norm is like an inner product, but since it has only one input
  we don't need the synckernel
*/
class omp_norm_kernel : virtual public omp_kernel {
private:
  distribution *local_scalar,*gathered_scalar;
  std::shared_ptr<object> local_value,squared;
protected:
  omp_kernel *prekernel,*sumkernel,*rootkernel;
public:
  omp_norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : omp_kernel(in,out),kernel(in,out),
      entity(entity_cookie::KERNEL)  {
    set_name("norm");

    // intermediate object for local sum:
    local_scalar = new omp_scalar_distribution(out.get());
    local_value = std::shared_ptr<object>( new omp_object(local_scalar) );
    local_value->set_name("local-inprod-value");
    
    // local norm kernel
    prekernel = new omp_kernel(in,local_value);
    prekernel->set_name("local-norm");
    prekernel->last_dependency()->set_explicit_beta_distribution(in.get()); 
    prekernel->set_localexecutefn( &local_normsquared );

    // reduction kernel
    squared = std::shared_ptr<object>( new omp_object( dynamic_cast<distribution*>(out.get()) ) );
    sumkernel = new omp_reduction_kernel(local_value,squared);

    // we need to take the root
    rootkernel = new omp_kernel(squared,out);
    rootkernel->set_name("root of squares");
    rootkernel->set_explicit_beta_distribution( squared.get() );
    rootkernel->set_localexecutefn( &vectorroot );
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(prekernel,sumkernel,rootkernel);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(prekernel,sumkernel,rootkernel);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() { prekernel->execute(); sumkernel->execute(); };
};

/*!
  Norm squared is much like norm. 
  \todo unify this with norm
*/
class omp_normsquared_kernel : virtual public omp_kernel {
private:
  distribution *local_scalar,*gathered_scalar;
  std::shared_ptr<object> local_value;
protected:
  omp_kernel *prekernel,*sumkernel;
public:
  omp_normsquared_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : omp_kernel(in,out),kernel(in,out),
      entity(entity_cookie::KERNEL)  {
    decomposition *decomp = out.get();
    set_name("norm squared");

    // intermediate object for local sums:
    local_scalar = new omp_scalar_distribution(out.get());
    local_value = std::shared_ptr<object>( new omp_object(local_scalar) );
    local_value->set_name("local-inprod-value");
    
    // local norm kernel
    prekernel = new omp_kernel(in,local_value);
    prekernel->set_name("local-norm");
    prekernel->last_dependency()->set_explicit_beta_distribution(in.get()); 
    prekernel->set_localexecutefn( &local_normsquared );
    //    prekernel->set_localexecutectx( (void*)in );

    // reduction kernel
    sumkernel = new omp_reduction_kernel(local_value,out);
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(prekernel,sumkernel);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(prekernel,sumkernel);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() override { prekernel->execute(); sumkernel->execute(); };
};

/*!
  Preconditioning is an unexplored topic

  \todo add a context, passing a sparse matrix
*/
class omp_preconditioning_kernel : public omp_kernel,public preconditioning_kernel {
public:
  omp_preconditioning_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),omp_kernel(in,out),preconditioning_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  An outer product kernel is a data parallel application of a redundantly 
  distributed array of length k with a disjointly distributed array of 
  size N, giving Nk points. The redundant array is stored as context,
  and the explosion function is passed as parameter.
*/
//snippet ompoutprodkernel
class omp_outerproduct_kernel : public omp_kernel {
public:
  omp_outerproduct_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,
			   std::shared_ptr<object> replicated,
			   void (*f)(kernel_function_types)
			   )
    : omp_kernel(in,out),entity(entity_cookie::KERNEL)  {
    set_name("outer product");
    set_localexecutefn(f);
    dependency *d = last_dependency();
    d->set_type_local();
  };
  //! \todo why isn't this inherited from omp_kernel
  task *make_task_for_domain(processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out) {
    return new omp_task(d,in,out); };
};
//snippet end

//! Trace kernel only has an input object
//! \todo why are we passing the string twice?
class omp_trace_kernel : virtual public omp_kernel,public trace_kernel {
public:
  omp_trace_kernel(std::shared_ptr<object> in,std::string c)
    : kernel(),omp_kernel(),trace_kernel(in,c),entity(entity_cookie::KERNEL) {
    out_object = std::shared_ptr<object>( new omp_object( dynamic_cast<distribution*>(in.get()) ) );
  };
};

//! An OMP central differences kernel is exactly the same as \ref centraldifference_kernel
class omp_centraldifference_kernel : public omp_kernel,public centraldifference_kernel {
public:
  omp_centraldifference_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),omp_kernel(in,out),centraldifference_kernel(in,out),
      entity(entity_cookie::KERNEL) {
  };
};

class omp_diffusion_kernel : virtual public omp_kernel {
public:
  omp_diffusion_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),omp_kernel(in,out),entity(entity_cookie::KERNEL) {
    add_sigma_operator( ioperator("none") ); // we need the i index
    add_sigma_operator( ioperator(">=1") );  // we need the i+1 index
    add_sigma_operator( ioperator("<=1") );  // we need the i-1 index
    double damp = 1./6;
    set_localexecutefn
      ( [damp] ( kernel_function_args ) -> void {
	return central_difference_damp( kernel_function_call,damp ); } );
  };
};

#endif
