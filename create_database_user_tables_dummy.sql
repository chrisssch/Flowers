create database Flowers
go

use Flowers
go

create login FlowersConnector with password='Pa$$w0rd'
create user FlowersConnector for login FlowersConnector
grant connect, select, insert, update to FlowersConnector

create table dbo.ImageTable(
	ImageID bigint identity(1,1) primary key -- automatically generated
	,ImageName nvarchar(255)
	,CreateDate datetime
)

create table dbo.ScoreTable(
	GenericID bigint identity(1,1) primary key -- automatically generated
	,ImageID bigint foreign key references ImageTable(ImageID) not null
	,Classifier nvarchar(255)
	,ClassifierType nvarchar(255)
	,Class nvarchar(255)
	,Score float
	,CreateDate datetime
)
