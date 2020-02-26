// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_ops.h: Header file for the operation kernels
 ****
 ****************************************************************/

#ifndef IMP_OPS_H
#define IMP_OPS_H 1

#include "imp_base.h"
#include "imp_functions.h"

/*! \page ops IMP Operations
  The basic mechanism of declaring distributions, objects, and kernels
  is powerful enough by itself. However, for convenience a number 
  of common operations have been declared.

  Somewhat remarkably, we define these operations independent
  of the parallelism mode. In files such as mpi_ops.h we then define
  the specific operation kernels by also letting them inherit
  from mpi_kernel and such.

*/


class setconstant_kernel : virtual public origin_kernel {
public:
  setconstant_kernel( std::shared_ptr<object> out,double v )
    : kernel(out),origin_kernel(out),entity(entity_cookie::KERNEL) {
    set_localexecutefn
      ( std::function< kernel_function_proto >{
	[v] ( kernel_function_args ) -> void {
	  vecsetconstant( kernel_function_call,v ); } } );
  };
};
  
//snippet impcopy
class copy_kernel : virtual public kernel {
public:
  copy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in copy kernel"));
    if (out==nullptr) throw(std::string("Null out object in copy kernel"));

    set_name(fmt::format("copy{}",get_out_object()->get_object_number()));
    dependency *d = last_dependency();
    d->set_explicit_beta_distribution(out.get());
    localexecutefn = &veccopy;
  };
};
//snippet end

//snippet impbcast
class bcast_kernel : virtual public kernel {
public:
  bcast_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in copy kernel"));
    if (out==nullptr) throw(std::string("Null out object in copy kernel"));

    set_name(fmt::format("bcast{}",get_out_object()->get_object_number()));
    dependency *d = last_dependency();
    d->set_explicit_beta_distribution(out.get());
    set_localexecutefn( &crudecopy );
  };
};
//snippet end

//snippet impgather
class gather_kernel : virtual public kernel {
private:
  std::shared_ptr<object> local_value,squared;
  int groupsize{4};
protected:
  kernel *prekernel,*scalar_gather;
public:
  gather_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out, std::function<kernel_function_proto> flocal )
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    set_cookie(entity_cookie::SHELLKERNEL);
    int step = out->get_object_number();
    set_name(fmt::format("norm{}",step));

    // intermediate object for local sum has same distribution as `out':
    {
      distribution *local_scalar = out->new_scalar_distribution();
      local_value = out->new_object(local_scalar);
    }
    local_value->set_name(fmt::format("local-norm-value{}",step));
    
    // local scalarization kernel
    prekernel = out->kernel_from_objects(in,local_value);
    prekernel->set_localexecutefn( &veccopy );
    prekernel->set_name(fmt::format("local-f-compute{}",step));
    prekernel->set_explicit_beta_distribution( in.get() );
    prekernel->set_localexecutefn(flocal );

    // scalar gather
    scalar_gather = out->kernel_from_objects(local_value,out);
    scalar_gather->set_name(fmt::format("scalar-gather-{}",step));
    scalar_gather->set_explicit_beta_distribution(out.get());
    scalar_gather->set_localexecutefn( &veccopy );
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(prekernel,scalar_gather);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(prekernel,scalar_gather);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() {
    prekernel->execute(); scalar_gather->execute();
  };
  int get_groupsize() { return groupsize; };
};
//snippet end

//! \todo the context can now go, change the void* parameter to double*
class axpy_kernel : virtual public kernel {
public:
  axpy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,double *x ) : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in axpy kernel"));
    if (out==nullptr) throw(std::string("Null out object in axpy kernel"));

    set_name(fmt::format("axpy{}",get_out_object()->get_object_number()));
    localexecutefn = [x] ( kernel_function_args) -> void {
      return vecscalebyc( kernel_function_call,*x ); };
    dependency *d = last_dependency();
    d->set_explicit_beta_distribution(out.get());
  };
};

class scale_kernel : virtual public kernel {
public:
  //! Scale kernel constructor; not to be called explicitly.
  scale_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in scale kernel"));
    if (out==nullptr) throw(std::string("Null out object in scale kernel"));

    set_name(fmt::format("scale {}->{}",in->get_object_number(),out->get_object_number()));
    set_localexecutefn( &vecscaleby );
    dependency *d = last_dependency();
    d->set_explicit_beta_distribution(out.get());
  };
  //! Scale by an explicit double. \todo the context can go now that we are capturing it
  scale_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out ) : scale_kernel(in,out) {
    set_name(fmt::format("scaleby{}",get_out_object()->get_object_number()));
    localexecutefn = [a] ( kernel_function_args ) -> void {
      return vecscalebyc( kernel_function_call,*a ); };
  };
  //snippet scalekernel
  scale_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out ) : scale_kernel(in,out) {
    set_name(fmt::format("scaleby object{}",get_out_object()->get_object_number()));
    // second in object is the scalar
    a->require_type_replicated();
    add_in_object(a);
    dependency *d = last_dependency(); d->set_name("wait for scalar");
    d->set_explicit_beta_distribution( a.get() );
  };
  //snippet end
};

//! Scale an object by the inverse of a provided scalar \todo define vectorvectorkernel for the "wait for in" part
//! \todo the context can go
class scaledown_kernel : virtual public kernel {
public:
  scaledown_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in scaledown kernel"));
    if (out==nullptr) throw(std::string("Null out object in scaledown kernel"));

    set_name(fmt::format("scaledownby scalar{}",get_out_object()->get_object_number()));
    dependency *d = last_dependency(); d->set_name("wait for in");
    d->set_explicit_beta_distribution(out.get());
    localexecutefn = [a] (kernel_function_args) -> void {
      return vecscaledownbyc( kernel_function_call,*a ); };
  };
  scaledown_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    if (in==nullptr) throw(std::string("Null in object in scaledown kernel"));
    if (out==nullptr) throw(std::string("Null out object in scaledown kernel"));

    set_name(fmt::format("scaledownby object{}",get_out_object()->get_object_number()));
    set_localexecutefn( &vecscaledownby );
    dependency *d = last_dependency(); d->set_name("wait for in");
    d->set_explicit_beta_distribution(out.get());
    // second in object is the scale
    a->require_type_replicated();
    add_in_object(a);
    d = last_dependency(); d->set_name("wait for scalar");
    d->set_explicit_beta_distribution(a.get()); //type_local();
  };
};

//snippet vecsum
class sum_kernel : virtual public kernel {
public:
  sum_kernel( std::shared_ptr<object> in1,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),entity(entity_cookie::KERNEL) {
    if (in1==nullptr) throw(std::string("Null in1 object in sum kernel"));
    if (in2==nullptr) throw(std::string("Null in2 object in sum kernel"));
    if (out==nullptr) throw(std::string("Null out object in sum kernel"));

    dependency *d;
    set_name(fmt::format("vector sum{}",get_out_object()->get_object_number()));
    d = last_dependency(); d->set_explicit_beta_distribution(out.get());
    // second vector should also be local
    add_in_object(in2);
    d = last_dependency(); d->set_explicit_beta_distribution(out.get());
    localexecutefn = &vectorsum;
  }
};
//snippet end

//! Operate on two replicated scalars.
//! \todo remove in2 from the chobst
class scalar_kernel : virtual public kernel {
public:
  scalar_kernel( std::shared_ptr<object> in1,const char *op,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),entity(entity_cookie::KERNEL) {
    if (in1==nullptr) throw(std::string("Null in1 object in scalar kernel"));
    if (in2==nullptr) throw(std::string("Null in2 object in scalar kernel"));
    if (out==nullptr) throw(std::string("Null out object in scalar kernel"));

    if (!in1->has_type_replicated())
      throw("Not replicated: in1\n");
    if (!in2->has_type_replicated())
      throw("Not replicated: in2\n");
    if (!out->has_type_replicated())
      throw("Not replicated: out\n");
    set_name(fmt::format("scalar-op{}",get_out_object()->get_object_number()));
    dependency *d = last_dependency(); d->set_type_local();
    add_in_object(in2); d = last_dependency(); d->set_type_local();
    //&char_scalar_op);
    char_object_struct *chobst = new char_object_struct;
    chobst->op = op; chobst->obj = in2;
    set_localexecutectx(chobst);
    set_localexecutefn
      ( [chobst] ( kernel_function_args ) -> void {
	return char_scalar_op( kernel_function_call,(void*)chobst ); } );
  };
};

/*!
  AXBYZ is a local kernel; the scalar is passed as the context. The scalar
  comes in as double*, this leaves open the possibility of an array of scalars,
  also it puts my mind at ease re that casting to void*.
  \todo now it's really a two-character struct
  \todo the explicit distributions on s1/s2 should really be gathered-whatever
*/
class axbyz_kernel : virtual public kernel {
protected:
public:
  axbyz_kernel( char op1,std::shared_ptr<object> s1,std::shared_ptr<object> x1,
		char op2,std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out )
    : kernel(x1,out),entity(entity_cookie::KERNEL) {
    if (s1->get_type()!=distribution_type::REPLICATED)
      throw(std::string("s1 not replicated"));
    if (s2->get_type()!=distribution_type::REPLICATED)
      throw(std::string("s2 not replicated"));
    dependency *d;
    set_name(fmt::format("ax+by=z{}",get_out_object()->get_object_number()));

    // x1 is local
    d = last_dependency(); d->set_name("wait x1"); d->set_explicit_beta_distribution(out.get());
    // s1 is local
    add_in_object(s1);
    d = last_dependency(); d->set_name("wait s1"); d->set_explicit_beta_distribution(s1.get());
    // x2 is local
    add_in_object(x2);
    d = last_dependency(); d->set_name("wait x2"); d->set_explicit_beta_distribution(out.get());
    // s2 is local
    add_in_object(s2);
    d = last_dependency(); d->set_name("wait s2"); d->set_explicit_beta_distribution(s2.get());

    // function and context   
    charcharxyz_object_struct *ctx = new charcharxyz_object_struct;
    ctx->c1 = op1; //ctx->s1 = s1;
    ctx->c2 = op2; //ctx->s2 = s2; //ctx->obj = x2;
    set_localexecutectx( ctx );
    localexecutefn = [ctx] (kernel_function_args) -> void {
      return vecaxbyz(kernel_function_call,(void*)ctx); };
  };
};

class norm_kernel : virtual public kernel {
private:
  std::shared_ptr<object> local_value,squared;
  int groupsize{4};
protected:
  kernel *prekernel,*sumkernel,*rootkernel;
public:
  //snippet normlocal
  norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    set_cookie(entity_cookie::SHELLKERNEL);
    int step = out->get_object_number();
    set_name(fmt::format("norm{}",get_out_object()->get_object_number()));

    // intermediate object for local sum has same distribution as `out':
    {
      distribution *local_scalar = out->new_scalar_distribution();
      local_value = out->new_object(local_scalar);
    }
    local_value->set_name(fmt::format("local-norm-value{}",step));
    
    // local norm kernel
    prekernel = out->kernel_from_objects(in,local_value);
    prekernel->set_name
      (fmt::format("local-norm-compute{}",get_out_object()->get_object_number()));
    prekernel->set_explicit_beta_distribution( in.get() );
    prekernel->set_localexecutefn( &local_normsquared );
    //snippet end

    // reduction kernel
    squared = out->new_object(out.get());
    squared->set_name(fmt::format("squared-inprod-value{}",step));
    sumkernel = kernel::make_reduction_kernel(local_value,squared); // new mpi_reduction_kernel

    // we need to take a root
    rootkernel = out->kernel_from_objects(squared,out);
    rootkernel->set_name(fmt::format("root of squares{}",get_out_object()->get_object_number()));
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
  virtual void execute() {
    prekernel->execute(); sumkernel->execute(); rootkernel->execute();
  };
  int get_groupsize() { return groupsize; };
};

class innerproduct_kernel : virtual public kernel {
private:
  distribution *local_scalar; // we need to keep them just to destroy them
  std::shared_ptr<object> local_value;
protected:
  kernel *prekernel,*sumkernel;
public:
  innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),entity(entity_cookie::KERNEL) {
    set_cookie(entity_cookie::SHELLKERNEL);

    int step = v2->get_object_number();
    set_name(fmt::format("inner-product{}",get_out_object()->get_object_number()));

    // intermediate object for local sum has same distribution as `out':
    try {
      distribution *local_scalar = global_sum->new_scalar_distribution();
      local_value = global_sum->new_object(local_scalar);
    } catch (std::string c) {
      throw(fmt::format("Inner product local scalar: {}",c)); }
    local_value->set_name(fmt::format("local-inprod-value{}",step));
    
    // local inner product kernel

    try { // v1 is in place      
      prekernel = global_sum->kernel_from_objects(v1,local_value);
      prekernel->set_name
	(fmt::format("local-innerproduct{}",get_out_object()->get_object_number()));
      prekernel->set_explicit_beta_distribution(v1.get());
      prekernel->last_dependency()->set_name("inprod wait for in vector");
    } catch (std::string c) {
      throw(fmt::format("Inprod prekernel v1: {}",c)); }

    try { // v2 is in place
      prekernel->add_in_object(v2);
      prekernel->set_explicit_beta_distribution(v2.get());
      prekernel->last_dependency()->set_name("inprod wait for second vector");
      prekernel->set_localexecutefn( &local_inner_product );
    } catch (std::string c) {
      throw(fmt::format("Inprod prekernel v2: {}",c)); }
    //fmt::print("Inprod prekernel has {} in objects\n",prekernel->get_dependencies().size());

    try { // reduction kernel
      sumkernel = kernel::make_reduction_kernel(local_value,global_sum);
    } catch (std::string c) {
      throw(fmt::format("Inprod sum kernel: {}",c)); }
  };
  /* ~mpi_innerproduct() { delete local_scalar; delete gathered_scalar; */
  /*   delete local_value; delete prekernel; delete synckernel; delete sumkernel; }; */
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks() override {
    split_contained_kernels(prekernel,sumkernel);
  };
  /*! We override the default analysis by analyzing in sequence the contained kernels.
    This splits the kernels and analyzes task dependencies; then we take the tasks
    and add them to the innerproduct task list.
  */
  virtual void analyze_dependencies() override {
    analyze_contained_kernels(prekernel,sumkernel);
    // for ( auto t : *get_tasks() )
    //   fmt::print("inprod task: {}\n",t->as_string());
  };
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() { // synckernel->execute();
    prekernel->execute(); sumkernel->execute(); };
  kernel *get_prekernel() { return prekernel; };
};

//! For now preconditioning is just a vector copy.
class preconditioning_kernel : virtual public kernel {
public:
  preconditioning_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    set_name(fmt::format("preconditioning{}",get_out_object()->get_object_number()));
    dependency *d = last_dependency();
    d->set_type_local();
    set_localexecutefn( &veccopy );
  };
};

//! One-dimensional central differences. One can later set another local function
//! to change the coefficients.
class centraldifference_kernel : virtual public kernel {
public:
  centraldifference_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),entity(entity_cookie::KERNEL) {
    last_dependency()->add_sigma_operator( ioperator("none") ); // we need the i index
    last_dependency()->add_sigma_operator( ioperator(">=1") );  // we need the i+1 index
    last_dependency()->add_sigma_operator( ioperator("<=1") );  // we need the i-1 index
    set_localexecutefn( &central_difference );
  };
};

/*
 * Tracing kernel
 */
class two_string_struct {
public:
  std::string string1,string2;
  two_string_struct( std::string s1,std::string s2 ) {
    string1 = s1; string2 = s2; };
};

/*! Trace kernel only has an in object; the out object is set anonymously
  in the mode-dependent derived class.
  The local method is \ref print_trace_message, which gets the format string
  as context.
*/
class trace_kernel : virtual public kernel {
public:
  trace_kernel( std::shared_ptr<object> in,std::string c )
    : kernel(),entity(entity_cookie::KERNEL) {
    set_name( std::string("trace kernel") );
    type = kernel_type::TRACE; add_in_object(in);
    
    std::string *cp = new std::string(c);
    last_dependency()->set_explicit_beta_distribution(in.get());
    localexecutefn = [cp] ( kernel_function_args ) -> void {
      return print_trace_message( kernel_function_call,(void*)cp ); };
    set_localexecutectx( cp );
  };
};

#endif
