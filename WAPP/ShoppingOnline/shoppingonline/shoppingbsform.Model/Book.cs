//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated from a template.
//
//     Manual changes to this file may cause unexpected behavior in your application.
//     Manual changes to this file will be overwritten if the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace shoppingbsform.Model
{
    using Dapper.Contrib.Extensions;
    using System;
    using System.Collections.Generic;

    [Table("[Book]")]
    public partial class Book
    {
        [ExplicitKey]
        public string Id { get; set; }
        public string Name { get; set; }
        public Nullable<decimal> Price { get; set; }
        public string Information { get; set; }
        public string Description { get; set; }
        public string Image { get; set; }
    }
}
