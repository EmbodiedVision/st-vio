// Generated by gencpp from file mavros_msgs/ManualControl.msg
// DO NOT EDIT!
/**
This file is auto-generated based on ManualControl.msg from
mavros https://github.com/mavlink/mavros
licensed under https://github.com/mavlink/mavros/blob/ros2/LICENSE.md
*/


#ifndef MAVROS_MSGS_MESSAGE_MANUALCONTROL_H
#define MAVROS_MSGS_MESSAGE_MANUALCONTROL_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace mavros_msgs
{
template <class ContainerAllocator>
struct ManualControl_
{
  typedef ManualControl_<ContainerAllocator> Type;

  ManualControl_()
    : header()
    , x(0.0)
    , y(0.0)
    , z(0.0)
    , r(0.0)
    , buttons(0)  {
    }
  ManualControl_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , x(0.0)
    , y(0.0)
    , z(0.0)
    , r(0.0)
    , buttons(0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef float _x_type;
  _x_type x;

   typedef float _y_type;
  _y_type y;

   typedef float _z_type;
  _z_type z;

   typedef float _r_type;
  _r_type r;

   typedef uint16_t _buttons_type;
  _buttons_type buttons;





  typedef boost::shared_ptr< ::mavros_msgs::ManualControl_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::mavros_msgs::ManualControl_<ContainerAllocator> const> ConstPtr;

}; // struct ManualControl_

typedef ::mavros_msgs::ManualControl_<std::allocator<void> > ManualControl;

typedef boost::shared_ptr< ::mavros_msgs::ManualControl > ManualControlPtr;
typedef boost::shared_ptr< ::mavros_msgs::ManualControl const> ManualControlConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::mavros_msgs::ManualControl_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::mavros_msgs::ManualControl_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::mavros_msgs::ManualControl_<ContainerAllocator1> & lhs, const ::mavros_msgs::ManualControl_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.x == rhs.x &&
    lhs.y == rhs.y &&
    lhs.z == rhs.z &&
    lhs.r == rhs.r &&
    lhs.buttons == rhs.buttons;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::mavros_msgs::ManualControl_<ContainerAllocator1> & lhs, const ::mavros_msgs::ManualControl_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace mavros_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::mavros_msgs::ManualControl_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::mavros_msgs::ManualControl_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mavros_msgs::ManualControl_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mavros_msgs::ManualControl_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mavros_msgs::ManualControl_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mavros_msgs::ManualControl_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::mavros_msgs::ManualControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "c41e3298484ea98e05ac502ce55af59f";
  }

  static const char* value(const ::mavros_msgs::ManualControl_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xc41e3298484ea98eULL;
  static const uint64_t static_value2 = 0x05ac502ce55af59fULL;
};

template<class ContainerAllocator>
struct DataType< ::mavros_msgs::ManualControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "mavros_msgs/ManualControl";
  }

  static const char* value(const ::mavros_msgs::ManualControl_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::mavros_msgs::ManualControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Manual Control state\n"
"std_msgs/Header header\n"
"float32 x\n"
"float32 y\n"
"float32 z\n"
"float32 r\n"
"uint16 buttons\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::mavros_msgs::ManualControl_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::mavros_msgs::ManualControl_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.x);
      stream.next(m.y);
      stream.next(m.z);
      stream.next(m.r);
      stream.next(m.buttons);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ManualControl_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::mavros_msgs::ManualControl_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::mavros_msgs::ManualControl_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "x: ";
    Printer<float>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<float>::stream(s, indent + "  ", v.y);
    s << indent << "z: ";
    Printer<float>::stream(s, indent + "  ", v.z);
    s << indent << "r: ";
    Printer<float>::stream(s, indent + "  ", v.r);
    s << indent << "buttons: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.buttons);
  }
};

} // namespace message_operations
} // namespace ros

#endif // MAVROS_MSGS_MESSAGE_MANUALCONTROL_H
