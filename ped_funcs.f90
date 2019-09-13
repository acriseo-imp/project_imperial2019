! Fortran versions of various Python functions in pedestrian code:
! spq (smallest_positive_quadroot)
! bypass_funcs
! f_w
! f_j
! f
! compute_alpha_des
! compute_d_h
! distance_to_wall

module ped_funcs
  implicit none
  real(kind=8), parameter :: pi = acos(-1.d0)
  real(kind=8), allocatable, dimension(:) :: pf_speed,pf_f_walls_array,pf_f_persons_array
  real(kind=8), allocatable, dimension(:,:) :: pf_walls, pf_contact_w !should be set in Python code
  contains
!-------------------------------------------------------------
!  Returns the smallest positive root of the quadratic equation
!  a*x^2 + b*x + c = 0.
!  If there is no positive root then -1 is returned
!  a2 = 2*a, d = b^2-4*a*c"""
subroutine spq(a2,b,d,root)
  real(kind=8), intent(in) :: a2,b,d
  real(kind=8), intent(out) :: root
  real(kind=8) :: b2a,sqrt_d,root1,root2

  b2a = -b/a2
  if (d<0) then
    root = -1
  elseif (d==0) then
    root = b2a
    if (root<=0) root=-1
  else
    sqrt_d = sqrt(d)
    root1 = (-b + sqrt_d) / (a2)
    root2 = (-b - sqrt_d) / (a2)

    if (root1 > 0) then
      if (root2>0) then
        if (root1 < root2) then
          root = root1
        else
          root = root2
        end if
      else
        root = root1
      end if
    elseif (root2>0) then
      root = root2
    else
      root = -1
    end if
  end if
end subroutine spq
!-------------------------------------------------------------

!-------------------------------------------------------------
subroutine bypass_funcs(dx_ij,dy_ij,alpha,d_max,d_by)
  implicit none
  real(kind=8), intent(in) :: dx_ij,dy_ij,alpha,d_max
  real(kind=8), intent(out) :: d_by
  real(kind=8) :: b_delta,b_direction,b1,b2

  b_delta = pi/2

  b_direction = atan2(-dy_ij,-dx_ij)

  b1 = modulo(b_direction-b_delta+pi,2*pi)-pi
  b2 = modulo(b_direction+b_delta+pi,2*pi)-pi

  if (alpha>b1 .or. alpha<b2) then
      d_by = 0
  else
      d_by = d_max
  end if

end subroutine bypass_funcs
!-------------------------------------------------------------

!-------------------------------------------------------------
!Compute the distance to collision with walls
subroutine f_w(alpha,i,wall,xi,yi,v_0i,ri,d_max,v_xi,v_yi,d_wall)
  implicit none
  integer, intent(in) :: i
  real(kind=8), intent(in) :: alpha,xi,yi,v_0i,ri,d_max,v_xi,v_yi
  real(kind=8), dimension(:), intent(in) :: wall
  real(kind=8), intent(out) :: d_wall

  real(kind=8) :: a,b,c,wall_start,wall_end,m,d,y_wall,delta_y,x_intercept
  real(kind=8) :: x_wall,delta_x,y_intercept,delta_tvals(2),r_component,abc_component
  real(kind=8) :: dist,f_alpha,delta_t
  integer :: flag

  a = wall(1)
  b = wall(2)
  c = wall(3)
  wall_start = wall(4) - ri
  wall_end = wall(5) + ri
  flag = 0

  if (abs(v_yi)>1e-14) then
    m = v_xi/v_yi
  elseif (v_xi>0) then
    m = 1e14
  else
    m = -1e14
  end if

  d = a*v_xi + b*v_yi
  if (d==0) then
    d_wall = 0
  else
    !horizontal wall
    if (a==0) then
      y_wall = -c/b
      if (v_yi==0) then
        d_wall = d_max
      elseif (v_yi>0) then
        if (yi>y_wall) then
          d_wall = d_max
        else
          delta_y = y_wall - (yi+ri)
          x_intercept = xi + delta_y*m

          if ((x_intercept<wall_start) .or. (x_intercept>wall_end)) then
            d_wall = d_max
          else
            flag = 1
          end if
        end if
      else
        if (yi<y_wall) then
          d_wall = d_max
        else
          delta_y = (yi-ri)-y_wall
          x_intercept = xi - delta_y*m
          if ((x_intercept<wall_start) .or. (x_intercept>wall_end)) then
            d_wall = d_max
          else
            flag = 1
          end if
        end if
      end if
    end if
    !end horizontal wall

    !vertical wall
    if (b==0) then
      if (abs(v_xi)>1e-14) then
        m = v_yi/v_xi
      elseif (v_yi>0) then
        m = 1e14
      else
        m = -1e14
      end if

      x_wall = -c/a
      if (v_xi==0) then
        d_wall = d_max
      elseif (v_xi>0) then
        if (xi>x_wall) then
          d_wall = d_max
        else
          delta_x = x_wall-(xi+ri)
          y_intercept = yi + delta_x*m
          if ((y_intercept<wall_start) .or. (y_intercept>wall_end)) then
            d_wall = d_max
          else
            flag = 1
          end if
        end if
      else
        if (xi < x_wall) then
          d_wall = d_max
        else
          delta_x = -x_wall+(xi-ri)
          y_intercept = yi - delta_x*m
          if ((y_intercept<wall_start) .or. (y_intercept>wall_end)) then
            d_wall = d_max
          else
            flag = 1
          end if
        end if
      end if
    end if
        !end vertical wall
  end if !if d==0

  if (flag==1) then
    r_component = ri*sqrt(a*a + b*b)
    abc_component = -a*xi - b*yi - c
    delta_tvals(1) = (r_component + abc_component)/d
    delta_tvals(2) = (-r_component + abc_component)/d

    if (maxval(delta_tvals) <0) then
      d_wall = d_max
    else
      if (minval(delta_tvals)>=0) then
        delta_t = minval(delta_tvals)
      else
        delta_t = maxval(delta_tvals)
      end if

      dist = v_0i * delta_t
      ! dist = min(pf_speed(i+1),v_0i) * delta_t
      f_alpha = min(dist,d_max)
      d_wall = f_alpha
    end if
  end if
end subroutine f_w
!-------------------------------------------------------------

!-------------------------------------------------------------
!Compute the distance to colision with person j
subroutine f_j(alpha,i,j,dx_ij,dy_ij,d_max,rsum_ij,v_0i,v_xj,v_yj,dv_x,dv_y, &
                quad_A2_j,quad_B_ij,quad_C2_ij,quad_D_ij,gap_ij,d_coll)
  implicit none
  real(kind=8), intent(in) :: alpha,dx_ij,dy_ij,d_max,rsum_ij,v_0i,v_xj,v_yj, &
                    dv_x,dv_y,quad_A2_j,quad_B_ij,quad_C2_ij,quad_D_ij,gap_ij
  integer, intent(in) :: i,j
  real(kind=8) :: delta_t,dist,f_alpha
  real(kind=8), intent(out) :: d_coll

  if (gap_ij < rsum_ij) then
    d_coll = 0
  elseif (gap_ij == rsum_ij) then
    call bypass_funcs(dx_ij,dy_ij,alpha,d_max,d_coll)

  elseif ((v_0i==0) .and. (v_xj==0) .and. (v_yj==0)) then
    d_coll = 0
  else
    call spq(quad_A2_j,quad_B_ij,quad_D_ij,delta_t)

    if (delta_t>0) then
      dist = v_0i*delta_t
      ! dist = min(pf_speed(i+1),v_0i)*delta_t
      d_coll = min(dist,d_max)
    else
      d_coll = d_max
    end if
  end if
end subroutine f_j
!-------------------------------------------------------------

!-------------------------------------------------------------
!Compute the minimum distance to collision in this direction
subroutine f(index,n,nw,alpha,i,rsum_i,dx_i,dy_i,quad_C2_i,in_field,v_xi,v_yi,dv_x,dv_y, &
              quad_A2,x_i,v,o_i,gap_i,d_max,v_0_i,r_i,alpha_0_i,ar,f_alpha)
  implicit none
  integer, intent(in) :: index,n,nw,i !---INPUT VARIABLES---!
  real(kind=8), intent(in) :: alpha,v_xi,v_yi,d_max,v_0_i,r_i,alpha_0_i,ar
  integer, dimension(:), intent(in) :: in_field
  real(kind=8), dimension(:), intent(in) :: rsum_i,dx_i,dy_i,quad_C2_i,dv_x,dv_y, &
                                            quad_A2,o_i,x_i,gap_i
  real(kind=8), dimension(:,:), intent(in) :: v
  real(kind=8), intent(out) :: f_alpha !---OUTPUT VARIABLE---!
  integer :: i1,j1,j,w1 !---LOCAL VARIABLES---!
  real(kind=8) :: fp,fw,f_persons,f_walls,d_des
  real(kind=8), dimension(n) :: quad_B_i,quad_D_i

  quad_B_i = 2*(dv_x*dx_i + dv_y*dy_i) !used in smallest_positive_quadroot
  quad_D_i = quad_B_i**2-quad_A2*quad_C2_i

  !Find distance to collisions with persons
  f_persons = d_max

  do j1=1,size(in_field)
    j = in_field(j1)+1
    if (i+1 /= j) then
      call f_j(alpha,i,j,dx_i(j),dy_i(j),d_max,rsum_i(j), &
                      v_0_i,v(1,j),v(2,j),dv_x(j),dv_y(j),quad_A2(j),quad_B_i(j), &
                      quad_C2_i(j),quad_D_i(j),gap_i(j),fp)
      f_persons = min(f_persons,fp)
    end if
  end do

  f_walls = d_max
  do w1=1,nw
    call f_w(alpha,i,pf_walls(:,w1),x_i(1),x_i(2),v_0_i,r_i,d_max,v_xi,v_yi,fw)
    f_walls = min(f_walls,fw)
  end do

  pf_f_walls_array(index) = f_walls
  pf_f_persons_array(index) = f_persons
  f_alpha = min(f_persons,f_walls)

  if (abs(alpha-alpha_0_i) <= ar/2) then
    d_des = sqrt((x_i(1) - o_i(1))**2 + (x_i(2)-o_i(2))**2)
    if (d_des < f_alpha) then
      f_alpha=d_max
    end if
  end if
end subroutine f
!-------------------------------------------------------------

!-------------------------------------------------------------
!Compute the minimum distance function to find alpha_des over the horizon of alpha values
subroutine compute_alpha_des(n,nw,i,rsum_i,dx_i,dy_i,quad_C2_i,in_field_i, &
                             x_i,v,o_i,gap_i,d_max,v_0_i,r_i,alpha_0_i,ar,alpha_current_i, &
                             H_i,alpha_out,f_alpha_out,d_h_out)
  !Computes distance from person i to wall w
  implicit none
  integer, intent(in) :: n,nw,i
  real(kind=8), intent(in) :: d_max,v_0_i,r_i,alpha_0_i,ar,alpha_current_i,H_i
  integer, dimension(:), intent(in) :: in_field_i
  real(kind=8), dimension(:), intent(in) :: rsum_i,dx_i,dy_i,quad_C2_i,o_i,x_i,gap_i
  real(kind=8), dimension(:,:), intent(in) :: v
  real(kind=8), intent(out) :: alpha_out,f_alpha_out,d_h_out

  integer :: i1,j1,w1,index,Nalpha,min_distance_index(1)
  real(kind=8) :: d_w,d_p
  real(kind=8), allocatable, dimension(:) :: alphas,v_xi_a,v_yi_a,f_alphas,distances
  real(kind=8), allocatable, dimension(:,:) :: dv_x,dv_y,quad_A2



  do w1 = 1,nw
    call distance_to_wall(x_i,w1,r_i,pf_contact_w(i+1,w1))
  end do

  if (nw>0) then
    where (pf_contact_w(i+1,:)>=r_i)
      pf_contact_w(i+1,:) = 0
    end where
  end if

  Nalpha = nint(2*H_i/ar)
  allocate(alphas(Nalpha),v_xi_a(Nalpha),v_yi_a(Nalpha),f_alphas(Nalpha),distances(Nalpha))
  allocate(dv_x(Nalpha,n),dv_y(Nalpha,n),quad_A2(Nalpha,n))

  do i1=1,Nalpha
    alphas(i1) = i1-1
  end do

  alphas = alpha_current_i-H_i + ar*alphas

  alphas = modulo(alphas+pi,2*pi)-pi

  if (allocated(pf_f_walls_array)) deallocate(pf_f_walls_array)
  allocate(pf_f_walls_array(Nalpha))

  if (allocated(pf_f_persons_array)) deallocate(pf_f_persons_array)
  allocate(pf_f_persons_array(Nalpha))

   v_xi_a = cos(alphas)*v_0_i
   v_yi_a = sin(alphas)*v_0_i

  ! v_xi_a = cos(alphas)*pf_speed(i+1)
  ! v_yi_a = sin(alphas)*pf_speed(i+1)

  do i1 = 1,n
    dv_x(:,i1) = v_xi_a - v(1,i1)
    dv_y(:,i1) = v_yi_a - v(2,i1)
  end do

  quad_A2 = 2*((dv_x)**2 + (dv_y)**2) !used in smallest_positive_quadroot

  f_alphas = 0
  do index = 1,Nalpha
    call f(index,n,nw,alphas(index),i,rsum_i,dx_i,dy_i,quad_C2_i,in_field_i, &
                        v_xi_a(index),v_yi_a(index),dv_x(index,:),dv_y(index,:),quad_A2(index,:), &
                        x_i,v,o_i,gap_i,d_max,v_0_i,r_i,alpha_0_i,ar,f_alphas(index))
  end do

  distances = d_max**2 + f_alphas**2 - 2*d_max*f_alphas*cos(alpha_0_i-alphas)

  min_distance_index = minloc(distances)
  alpha_out = alphas(min_distance_index(1))
  f_alpha_out = f_alphas(min_distance_index(1))
  d_w = pf_f_walls_array(min_distance_index(1)) !shortest distance to a wall along direction alpha_out
  d_p = pf_f_persons_array(min_distance_index(1)) !shortest distance to a moving person along direction alpha_out


  call compute_d_h(n,i,dx_i,dy_i,alpha_out,quad_C2_i,in_field_i,d_w,d_p,d_h_out)
!  print *, "i,alpha,f_alpha,d_w,d_p,d_h=",i,alpha_out,f_alpha_out,d_w,d_p,d_h_out


end subroutine compute_alpha_des
!-------------------------------------------------------------
!compute shortest distance to collision along angle alpha_out
subroutine compute_d_h(n,i,dx_i,dy_i,alpha_out,quad_C2_i,in_field_i,d_w,d_p,d_h_out)
  implicit none
  integer, intent(in) :: n,i
  real(kind=8), dimension(:), intent(in) :: dx_i,dy_i,quad_C2_i
  integer, dimension(:), intent(in) :: in_field_i
  real(kind=8), intent(in) :: alpha_out,d_w,d_p
  real(kind=8), intent(out) :: d_h_out
  integer :: i1,j1,j
  real(kind=8) :: d_p_new,delta_t
  real(kind=8), dimension(n) :: quad_B_i,quad_D_i

  quad_B_i = 2*(cos(alpha_out)*dx_i + sin(alpha_out)*dy_i) !used in smallest_positive_quadroot
  quad_D_i = quad_B_i**2-2*quad_C2_i

  !compute d_p
  d_p_new = 100000.d0

  !check for contact
  if (d_p==0) then
    d_p_new = 0
  else
    !loop through in_field
    do j1=1,size(in_field_i)
      j = in_field_i(j1)+1
      !compute spq
        if (i+1 /= j) then
          call spq(2.d0,quad_B_i(j),quad_D_i(j),delta_t)
          if (delta_t>=0) d_p_new = min(d_p_new,delta_t)
        end if
    end do
  end if
  d_h_out = min(d_p_new,d_w)

end subroutine compute_d_h
!-------------------------------------------------------------
subroutine distance_to_wall(x_i,iw,ri,dist)
  implicit none
  real(kind=8), dimension(:), intent(in) :: x_i
  integer, intent(in) :: iw
  real(kind=8), intent(in) :: ri
  real(kind=8), intent(out) :: dist
  real(kind=8) :: xi,yi,a,b,c,wall_start,wall_end
  real(kind=8) :: x1,x2,y1,y2,tx,ty,val,x_val,y_val,dx,dy

  xi = x_i(1)
  yi = x_i(2)

  a = pf_walls(1,iw)
  b = pf_walls(2,iw)
  c = pf_walls(3,iw)
  wall_start = pf_walls(4,iw)
  wall_end = pf_walls(5,iw)

  if (b /= 0) then
    x1 = wall_start
    x2 = wall_end
    y1 = -(a*x1+c)/b
    y2 = -(a*x2+c)/b
  else
    x1 = -c/a
    x2 = -c/a
    y1 = wall_start
    y2 = wall_end
  end if

  tx = x2-x1
  ty = y2-y1

  val =  ((xi - x1) * tx + (yi - y1) * ty) / (tx*tx + ty*ty)
  if (val > 1) then
      val = 1
  elseif (val < 0) then
      val = 0
  end if
  x_val = x1 + val * tx
  y_val = y1 + val * ty
  dx = x_val - xi
  dy = y_val - yi
  dist = sqrt(dx*dx +dy*dy)
end subroutine distance_to_wall
!-------------------------------------------------------------

end module ped_funcs
!-------------------------------------------------------------
